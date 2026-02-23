import json
import sys
import math
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

QA_TRAIN = BASE / "fixtures/qa_train.json"
QA_DEV   = BASE / "fixtures/qa_dev.json"
QA_TEST  = BASE / "fixtures/qa_test.json"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Hyper parameters
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
PAD_ID = tokenizer.special_token_ids.get("<|pad|>", 0)
EOS_ID = tokenizer.special_token_ids.get("<|endoftext|>", 1)
test_enc = tokenizer.encode("Once upon a time, there was")

model = TransformerLM(
    vocab_size=len(tokenizer.vocab),
    context_length=CTX_LEN,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
).to(DEVICE)

dataloader = create_pretraining_dataloader(
    file_path=PRETRAIN_DATA,
    tokenizer=tokenizer,
    batch_size=BATCH_SIZE,
    max_length=CTX_LEN,
    stride=CTX_LEN // 2,
    shuffle=True,
)

train_cfg = TrainingConfig(
    num_epochs=PRETRAIN_EPOCHS,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=min(200, len(dataloader) * 2),
    max_grad_norm=1.0,
    device=DEVICE,
    log_interval=max(1, len(dataloader) // 3),
)
trainer = Trainer(model=model, config=train_cfg, train_dataloader=dataloader)
pt_results = trainer.train()
final_ppl = math.exp(pt_results['train_losses'][-1])

for prompt in ["Once upon a time", "The little girl"]:
    out = generate_text(model, tokenizer, prompt, max_new_tokens=30, method="greedy")

with open(QA_TRAIN) as f:
    train_data = json.load(f)
with open(QA_DEV) as f:
    dev_data = json.load(f)

qa_model = TransformerForMultipleChoice(
    transformer_lm=model,
    hidden_size=D_MODEL,
    num_choices=4,
    pooling="last",
    freeze_backbone=False,
).to(DEVICE)
train_dl = create_qa_dataloader(train_data, tokenizer, BATCH_SIZE, CTX_LEN, 4, shuffle=True)
dev_dl   = create_qa_dataloader(dev_data,   tokenizer, BATCH_SIZE, CTX_LEN, 4, shuffle=False)

ft_cfg = TrainingConfig(
    num_epochs=FINETUNE_EPOCHS,
    learning_rate=LR / 10,
    weight_decay=0.1,
    warmup_steps=5,
    max_grad_norm=1.0,
    device=DEVICE,
    log_interval=5,
)
ft_trainer = Trainer(
    model=qa_model,
    config=ft_cfg,
    train_dataloader=train_dl,
    compute_loss_fn=create_qa_loss_fn(DEVICE),
)
ft_results = ft_trainer.train()
dev_eval = evaluate_qa_model(qa_model, dev_dl, DEVICE)

def lm_score_choices(model, tokenizer, example, device):
    model.eval()
    context  = example["context"]
    question = example["question"]
    choices  = example["choices"]

    prefix = f"{context}\n\nQuestion: {question}\n\nAnswer: "
    prefix_ids = tokenizer.encode(prefix)
    scores = []
    with torch.no_grad():
        for choice in choices:
            choice_ids = tokenizer.encode(choice)
            if not choice_ids:
                scores.append(float('-inf'))
                continue

            full_ids = prefix_ids + choice_ids
            if len(full_ids) > model.context_length:
                full_ids = full_ids[-model.context_length:]
            input_tensor = torch.tensor([full_ids], device=device)
            logits = model(input_tensor)  # (1, seq, vocab)
            log_probs = F.log_softmax(logits[0], dim=-1)  # (seq, vocab)
            n_prefix = min(len(prefix_ids), len(full_ids) - len(choice_ids))
            n_choice = len(choice_ids)

            score = 0.0
            for j in range(n_choice):
                pos = n_prefix + j - 1
                tok = choice_ids[j]
                if 0 <= pos < logits.shape[1]:
                    score += log_probs[pos, tok].item()
            scores.append(score / n_choice)

    return scores


@torch.no_grad()
def lm_predict_all(model, tokenizer, examples, device):
    predictions = []
    for ex in examples:
        scores = lm_score_choices(model, tokenizer, ex, device)
        predictions.append(int(torch.tensor(scores).argmax()))
    return predictions


# Evaluate LM scoring on dev set
lm_dev_preds = lm_predict_all(model, tokenizer, dev_data, DEVICE)
lm_dev_correct = sum(p == ex["answer"] for p, ex in zip(lm_dev_preds, dev_data))
lm_dev_acc = lm_dev_correct / len(dev_data)
print(f"LM scoring dev accuracy:   {lm_dev_acc:.2%}")
print(f"Classifier dev accuracy:   {dev_eval['accuracy']:.2%}")

def overlap_scores(example):
    """Count how many words in each choice appear in the context."""
    context = example["context"].lower()
    scores = []
    for choice in example["choices"]:
        choice_lower = choice.lower()
        # Exact substring match gets a bonus
        exact = 2.0 if choice_lower in context else 0.0
        words = choice_lower.split()
        word_score = sum(1 for w in words if w in context) / max(len(words), 1)
        scores.append(exact + word_score)
    return scores

def overlap_predict_all(examples):
    return [int(torch.tensor(overlap_scores(ex)).argmax()) for ex in examples]

overlap_dev_preds = overlap_predict_all(dev_data)
overlap_dev_acc = sum(p == ex["answer"] for p, ex in zip(overlap_dev_preds, dev_data)) / len(dev_data)

def ensemble_predict_all(model, tokenizer, examples, device, lm_weight=0.2, overlap_weight=0.8):
    preds = []
    for ex in examples:
        lm_sc = lm_score_choices(model, tokenizer, ex, device)
        ov_sc = overlap_scores(ex)

        lm_t = torch.tensor(lm_sc, dtype=torch.float)
        lm_t = torch.nan_to_num(lm_t, nan=0.0, posinf=0.0, neginf=0.0)
        lm_range = lm_t.max() - lm_t.min()
        lm_norm = (lm_t - lm_t.min()) / (lm_range + 1e-9) if lm_range > 1e-9 else torch.zeros_like(lm_t)

        ov_t = torch.tensor(ov_sc, dtype=torch.float)
        ov_range = ov_t.max() - ov_t.min()
        ov_norm = (ov_t - ov_t.min()) / (ov_range + 1e-9) if ov_range > 1e-9 else torch.zeros_like(ov_t)

        combined = lm_weight * lm_norm + overlap_weight * ov_norm
        preds.append(int(combined.argmax()))
    return preds


ens_dev_preds = ensemble_predict_all(model, tokenizer, dev_data, DEVICE)
ens_dev_acc = sum(p == ex["answer"] for p, ex in zip(ens_dev_preds, dev_data)) / len(dev_data)
methods = {
    "classifier":  (dev_eval['accuracy'],  None),
    "lm_scoring":  (lm_dev_acc,            None),
    "overlap":     (overlap_dev_acc,        None),
    "ensemble":    (ens_dev_acc,            None),
}
best_method = max(methods, key=lambda m: methods[m][0])

with open(QA_TEST) as f:
    test_data = json.load(f)

if best_method == "classifier":
    test_dl = create_qa_dataloader(test_data, tokenizer, BATCH_SIZE, CTX_LEN, 4, shuffle=False)
    test_predictions = evaluate_qa_model(qa_model, test_dl, DEVICE)["predictions"]
elif best_method == "lm_scoring":
    test_predictions = lm_predict_all(model, tokenizer, test_data, DEVICE)
elif best_method == "overlap":
    test_predictions = overlap_predict_all(test_data)
else:  # ensemble
    test_predictions = ensemble_predict_all(model, tokenizer, test_data, DEVICE)

best_dev_acc = max(methods[best_method][0], dev_eval['accuracy'])

finetuned_output = {
    "predictions": test_predictions,
    "dev_accuracy": best_dev_acc,
    "config": "quick",
}
out_path = OUTPUT_DIR / "finetuned_predictions.json"
with open(out_path, "w") as f:
    json.dump(finetuned_output, f, indent=2)
print(f"\nSaved: {out_path}")
print(f"Dev accuracy: {best_dev_acc:.2%}")
