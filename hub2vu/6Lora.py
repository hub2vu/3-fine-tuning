# 이전 결과 삭제(재실행 대비)
shutil.rmtree(LORA_DIR, ignore_errors=True)

# 베이스 모델: 동일
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# 로베르타 계열의 어텐션 모듈명(query/key/value/dense)에 LoRA 적용
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value", "dense"],
    bias="none",
)

lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()  # 몇 % 파라미터가 학습되는지 확인용

lora_args = TrainingArguments(
    output_dir=LORA_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    warmup_ratio=WARMUP_RATIO,
    fp16=fp16,
    report_to="none",
    seed=SEED,
)

lora_trainer = Trainer(
    model=lora_model,
    args=lora_args,
    train_dataset=enc_train,
    eval_dataset=enc_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

reset_gpu_peak()
t0 = time.time()
print("== LoRA Training Started ==")
lora_train_result = lora_trainer.train()
t1 = time.time()

lora_train_seconds = t1 - t0
lora_val = lora_trainer.evaluate(enc_val)
lora_test = lora_trainer.evaluate(enc_test)
lora_gpu_peak = gpu_mem_peak_bytes()

# 저장(LoRA 어댑터 + 헤드만 저장됨)
lora_trainer.save_model(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)

lora_size = get_dir_size(LORA_DIR)

print("\n== LoRA Results ==")
print("Validation:", lora_val)
print("Test      :", lora_test)
print("Train time:", str(timedelta(seconds=int(lora_train_seconds))))
print("GPU peak  :", human_size(lora_gpu_peak))
print("Model dir :", LORA_DIR, human_size(lora_size))
print("\n[nvidia-smi snapshot]")
print_nvidia_smi()





# =========================
# LoRA 어댑터 로드 오류(dense) 패치 + 테스트 전체 코드
# =========================

# ---- 설정 ----
MODEL_NAME = "klue/roberta-base"
LORA_DIR   = "./ynat_runs/lora_roberta"
MAX_LENGTH = 128
SEED       = 42

import os, json, warnings
warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)
from sklearn.metrics import accuracy_score

set_seed(SEED)
device = 0 if torch.cuda.is_available() else -1

# ---- metrics ----
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ---- YNAT: test 분리 구성 ----
def build_test_enc(tokenizer, max_len=MAX_LENGTH, seed=SEED):
    raw = load_dataset("klue", "ynat")
    label_names = raw["train"].features["label"].names
    label2id = {n:i for i,n in enumerate(label_names)}
    id2label = {i:n for n,i in label2id.items()}

    def preprocess(batch):
        out = tokenizer(batch["title"], truncation=True, max_length=max_len)
        out["labels"] = batch["label"]
        return out

    split = raw["train"].train_test_split(test_size=0.1, seed=seed, stratify_by_column="label")
    ds_test = split["test"]
    enc_test = ds_test.map(preprocess, batched=True, remove_columns=ds_test.column_names)
    enc_test.set_format(type="torch")
    return enc_test, label_names, id2label, label2id

# ---- 토크나이저 로드 ----
try:
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR, use_fast=False)

enc_test, label_names, id2label, label2id = build_test_enc(tokenizer)
num_labels = len(label_names)

# ---- 1) adapter_config.json 패치: target_modules에서 'dense' 제거 ----
cfg_path = os.path.join(LORA_DIR, "adapter_config.json")
if os.path.exists(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    tmods = cfg.get("target_modules", [])
    if any(m == "dense" for m in tmods):
        cfg["target_modules"] = [m for m in tmods if m != "dense"]
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print("[PATCH] Removed 'dense' from target_modules in adapter_config.json:", tmods, "->", cfg["target_modules"])
    else:
        print("[INFO] adapter_config.json target_modules:", tmods)
else:
    print("[WARN] adapter_config.json not found at", cfg_path)

# ---- 2) 베이스 + 어댑터 로드 (충돌 방지) ----
from peft import PeftModel
base = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# 로딩 시, 패치된 target_modules에 맞춰 어댑터 주입
model = PeftModel.from_pretrained(base, LORA_DIR)
model.eval()
print("[INFO] LoRA adapter loaded on base model.")

# ---- 3) 평가 ----
args = TrainingArguments(
    output_dir="./tmp_eval_lora",
    report_to="none",
    per_device_eval_batch_size=64
)
trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=enc_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("== LoRA Test Metrics ==")
print(trainer.evaluate(enc_test))

# ---- 4) 샘플 예측(라벨 + 확률 %) ----
from transformers import pipeline
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,
    top_k=None
)

samples = [
    "국회, 내년도 예산안 쟁점 조율… 여야 막판 협상",
    "애플, 최신 아이패드 공개… 신형 칩셋·배터리 개선",
    "세계 선수권에서 한국 양궁 대표팀 금메달 획득",
    "금리 동결에 주식·채권 시장 혼조세",
]

print("\n== LoRA Predictions ==")
for s in samples:
    out = clf(s)
    # 다양한 반환 포맷 방어
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        pred = out[0]
    elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        pred = out[0][0]
    else:
        print("- raw:", out); continue

    label = pred.get("label", "N/A")
    score = float(pred.get("score", 0.0)) * 100.0
    print(f"- {label} ({score:.1f}%)")
