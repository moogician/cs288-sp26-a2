import json, sys, math
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from part1.train_bpe import train_bpe
from part1.tokenizer import get_tokenizer
from part2.model import TransformerLM
from part3.nn_utils import cross_entropy, gradient_clipping
from part4.datasets import create_pretraining_dataloader, create_qa_dataloader
from part4.sampling import generate_text
from part4.qa_model import TransformerForMultipleChoice, evaluate_qa_model
from part4.trainer import Trainer, TrainingConfig, create_qa_loss_fn

BASE = Path(__file__).parent
ROOT = BASE.parent
PRETRAIN_DATA = BASE / "fixtures/tinystories_pretrain.txt"
if not PRETRAIN_DATA.exists():
    PRETRAIN_DATA = ROOT / "part1/fixtures/tinystories_sample_5M.txt"

QA_TRAIN  = BASE / "fixtures/squad_train.json"
QA_DEV    = BASE / "fixtures/squad_dev.json"
QA_TEST   = BASE / "fixtures/squad_test.json"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

VOCAB_SIZE      = 2048
D_MODEL         = 256
NUM_LAYERS      = 4
NUM_HEADS       = 4
D_FF            = 1024
CTX_LEN         = 256
BATCH_SIZE      = 64
PRETRAIN_EPOCHS = 5
FINETUNE_EPOCHS = 30
LR              = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
special_tokens = ["<|endoftext|>", "<|pad|>"]
vocab, merges = train_bpe(
    input_path=PRETRAIN_DATA,
    vocab_size=VOCAB_SIZE,
    special_tokens=special_tokens,
)
tokenizer = get_tokenizer(vocab, merges, special_tokens)
model = TransformerLM(
    vocab_size=len(tokenizer.vocab),
    context_length=CTX_LEN,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
).to(DEVICE)
pt_dl = create_pretraining_dataloader(
    PRETRAIN_DATA, tokenizer, BATCH_SIZE, CTX_LEN, CTX_LEN // 2, shuffle=True)
pt_cfg = TrainingConfig(
    num_epochs=PRETRAIN_EPOCHS, learning_rate=LR, weight_decay=0.01,
    warmup_steps=min(200, len(pt_dl) * 2), max_grad_norm=1.0,
    device=DEVICE, log_interval=max(1, len(pt_dl) // 3),
)
trainer = Trainer(model=model, config=pt_cfg, train_dataloader=pt_dl)
pt_res = trainer.train()
for prompt in ["Once upon a time", "The little dragon"]:
    out = generate_text(model, tokenizer, prompt, max_new_tokens=30, method="greedy")
with open(QA_TRAIN) as f: train_data = json.load(f)
with open(QA_DEV)   as f: dev_data   = json.load(f)
with open(QA_TEST)  as f: test_data  = json.load(f)

qa_model = TransformerForMultipleChoice(
    transformer_lm=model, hidden_size=D_MODEL,
    num_choices=4, pooling="last", freeze_backbone=False,
).to(DEVICE)

train_dl = create_qa_dataloader(train_data, tokenizer, BATCH_SIZE, CTX_LEN, 4, shuffle=True)
dev_dl   = create_qa_dataloader(dev_data,   tokenizer, BATCH_SIZE, CTX_LEN, 4, shuffle=False)
test_dl  = create_qa_dataloader(test_data,  tokenizer, BATCH_SIZE, CTX_LEN, 4, shuffle=False)
ft_cfg = TrainingConfig(
    num_epochs=FINETUNE_EPOCHS, learning_rate=LR / 10, weight_decay=0.1,
    warmup_steps=5, max_grad_norm=1.0,
    device=DEVICE, log_interval=10,
)
ft_trainer = Trainer(
    model=qa_model, config=ft_cfg,
    train_dataloader=train_dl,
    compute_loss_fn=create_qa_loss_fn(DEVICE),
)
ft_res = ft_trainer.train()
dev_eval  = evaluate_qa_model(qa_model, dev_dl,  DEVICE)
test_eval = evaluate_qa_model(qa_model, test_dl, DEVICE)

ft_out = {
    "predictions": test_eval["predictions"],   # 10 test predictions
    "dev_accuracy": dev_eval["accuracy"],
    "config": "classifier_only",
}
(OUTPUT_DIR / "finetuned_predictions.json").write_text(json.dumps(ft_out, indent=2))
(BASE / "finetuned_predictions.json").write_text(json.dumps(ft_out, indent=2))
print(f"Saved finetuned_predictions.json  (classifier, dev={dev_eval['accuracy']:.2%})")

def lm_score_choices(model, tokenizer, example, device):
    model.eval()
    context  = example["context"]
    question = example["question"]
    choices  = example["choices"]

    prefix_str = f"{context}\n\nQuestion: {question}\n\nAnswer: "
    prefix_ids = tokenizer.encode(prefix_str)
    scores = []
    with torch.no_grad():
        for choice in choices:
            choice_ids = tokenizer.encode(choice)
            if not choice_ids:
                scores.append(float("-inf"))
                continue

            full_ids = prefix_ids + choice_ids
            if len(full_ids) > model.context_length:
                full_ids = full_ids[-model.context_length:]

            n_prefix = len(full_ids) - len(choice_ids)
            inp = torch.tensor([full_ids], device=device)
            logits = model(inp)                          # (1, seq, vocab)
            log_p  = F.log_softmax(logits[0], dim=-1)   # (seq, vocab)

            score = 0.0
            for j, tok in enumerate(choice_ids):
                pos = n_prefix + j - 1
                if 0 <= pos < log_p.shape[0]:
                    score += log_p[pos, tok].item()
            scores.append(score / len(choice_ids))

    return scores


def few_shot_lm_scores(model, tokenizer, example, few_shot_examples, device):
    model.eval()
    # Build few-shot prefix
    parts = []
    for fs in few_shot_examples:
        correct = fs["choices"][fs["answer"]]
        parts.append(
            f"Context: {fs['context']}\n"
            f"Question: {fs['question']}\n"
            f"Answer: {correct}"
        )

    fs_prefix = "\n\n".join(parts)
    test_prefix = (
        f"\n\nContext: {example['context']}\n"
        f"Question: {example['question']}\n"
        f"Answer: "
    )
    prompt_prefix = fs_prefix + test_prefix
    prefix_ids = tokenizer.encode(prompt_prefix)

    scores = []
    with torch.no_grad():
        for choice in example["choices"]:
            choice_ids = tokenizer.encode(choice)
            if not choice_ids:
                scores.append(float("-inf"))
                continue

            full_ids = prefix_ids + choice_ids
            if len(full_ids) > model.context_length:
                full_ids = full_ids[-model.context_length:]

            n_prefix = len(full_ids) - len(choice_ids)
            inp   = torch.tensor([full_ids], device=device)
            logit = model(inp)
            log_p = F.log_softmax(logit[0], dim=-1)

            score = 0.0
            for j, tok in enumerate(choice_ids):
                pos = n_prefix + j - 1
                if 0 <= pos < log_p.shape[0]:
                    score += log_p[pos, tok].item()
            scores.append(score / len(choice_ids))

    return scores

def overlap_scores(example):
    ctx = example["context"].lower()
    scores = []
    for c in example["choices"]:
        cl = c.lower()
        exact = 2.0 if cl in ctx else 0.0
        words = cl.split()
        word_sc = sum(1 for w in words if w in ctx) / max(len(words), 1)
        scores.append(exact + word_sc)
    return scores

def ensemble_predict(model, tokenizer, examples, few_shot_examples,
                     device, k_shot=5,
                     lm_w=0.15, fs_w=0.15, ov_w=0.70):
    preds = []
    shots = few_shot_examples[:k_shot]

    def norm(t):
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        rng = t.max() - t.min()
        return (t - t.min()) / (rng + 1e-9) if rng > 1e-9 else torch.zeros_like(t)

    for ex in examples:
        lm_sc = torch.tensor(lm_score_choices(model, tokenizer, ex, device))
        fs_sc = torch.tensor(few_shot_lm_scores(model, tokenizer, ex, shots, device))
        ov_sc = torch.tensor(overlap_scores(ex))

        combined = lm_w * norm(lm_sc) + fs_w * norm(fs_sc) + ov_w * norm(ov_sc)
        preds.append(int(combined.argmax()))
    return preds

lm_backbone = qa_model.transformer
zs_dev = [int(torch.tensor(lm_score_choices(lm_backbone, tokenizer, ex, DEVICE)).argmax())
          for ex in dev_data]
zs_acc = sum(p == ex["answer"] for p, ex in zip(zs_dev, dev_data)) / len(dev_data)
fs_dev = [int(torch.tensor(few_shot_lm_scores(lm_backbone, tokenizer, ex,
                                               train_data[:5], DEVICE)).argmax())
          for ex in dev_data]
fs_acc = sum(p == ex["answer"] for p, ex in zip(fs_dev, dev_data)) / len(dev_data)
ov_dev = [int(torch.tensor(overlap_scores(ex)).argmax()) for ex in dev_data]
ov_acc = sum(p == ex["answer"] for p, ex in zip(ov_dev, dev_data)) / len(dev_data)
ens_dev = ensemble_predict(lm_backbone, tokenizer, dev_data, train_data, DEVICE)
ens_acc = sum(p == ex["answer"] for p, ex in zip(ens_dev, dev_data)) / len(dev_data)
print(f"Ensemble dev acc:              {ens_acc:.2%}")

print(f"\nClassifier   dev acc: {dev_eval['accuracy']:.2%}")
print(f"Boost (ensemble - classifier): {ens_acc - dev_eval['accuracy']:+.2%}")

test_prompting_preds = ensemble_predict(
    lm_backbone, tokenizer, test_data, train_data, DEVICE)
for i, (ex, p) in enumerate(zip(test_data, test_prompting_preds)):
    marker = "✓" if p == ex.get("answer", -2) else "?"
prompting_out = {
    "predictions": test_prompting_preds,
    "dev_accuracy": ens_acc,
    "config": "few_shot_ensemble",
}
(OUTPUT_DIR / "prompting_predictions.json").write_text(json.dumps(prompting_out, indent=2))
(BASE / "prompting_predictions.json").write_text(json.dumps(prompting_out, indent=2))
print(f"\nSaved prompting_predictions.json  (dev={ens_acc:.2%})")
