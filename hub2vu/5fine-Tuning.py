# 이전 결과 삭제(재실행 대비)
shutil.rmtree(BASELINE_DIR, ignore_errors=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

fp16 = torch.cuda.is_available()  # GPU면 자동으로 fp16 켜기
args = TrainingArguments(
    output_dir=BASELINE_DIR,
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

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=enc_train,
    eval_dataset=enc_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

reset_gpu_peak()
t0 = time.time()
print("== Baseline Training Started ==")
train_result = trainer.train()
t1 = time.time()

baseline_train_seconds = t1 - t0
baseline_val = trainer.evaluate(enc_val)
baseline_test = trainer.evaluate(enc_test)
baseline_gpu_peak = gpu_mem_peak_bytes()

# 저장
trainer.save_model(BASELINE_DIR)
tokenizer.save_pretrained(BASELINE_DIR)

baseline_size = get_dir_size(BASELINE_DIR)

print("\n== Baseline Results ==")
print("Validation:", baseline_val)
print("Test      :", baseline_test)
print("Train time:", str(timedelta(seconds=int(baseline_train_seconds))))
print("GPU peak  :", human_size(baseline_gpu_peak))
print("Model dir :", BASELINE_DIR, human_size(baseline_size))
print("\n[nvidia-smi snapshot]")
print_nvidia_smi()




# fine-tuning 모델 테스트 (평가 + 예측)
# =========================

# ==== 설정 ====
MODEL_NAME   = "klue/roberta-base"
BASELINE_DIR = "./ynat_runs/baseline_roberta"
MAX_LENGTH   = 128
SEED         = 42

import os, warnings
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
set_seed(SEED)

device = 0 if torch.cuda.is_available() else -1

# ===== compute_metrics =====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits: (N, num_labels), labels: (N,)
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ===== enc_test가 없으면 재구성 =====
def ensure_test_enc(tokenizer, max_len=MAX_LENGTH, seed=SEED):
    """
    - KLUE/YNAT: 공개 test split이 없으므로 train에서 10% stratified split으로 test 생성
    - enc_test만 만들고, 평가에만 사용
    """
    raw = load_dataset("klue", "ynat")
    label_names = raw["train"].features["label"].names
    label2id = {n:i for i,n in enumerate(label_names)}
    id2label = {i:n for n,i in label2id.items()}

    def preprocess(batch):
        out = tokenizer(batch["title"], truncation=True, max_length=max_len)
        out["labels"] = batch["label"]
        return out

    split = raw["train"].train_test_split(
        test_size=0.1, seed=seed, stratify_by_column="label"
    )
    ds_test = split["test"]
    enc_test = ds_test.map(preprocess, batched=True, remove_columns=ds_test.column_names)
    enc_test.set_format(type="torch")
    return enc_test, label_names, id2label, label2id

# ===== 로드 & enc_test 준비 =====
# 토크나이저/모델은 학습 저장 디렉토리에서 로드 (라벨 매핑 일관성 보장)
try:
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_DIR, use_fast=True)
except Exception:
    # 혹시 fast tokenizer가 환경에서 문제되면 slow로 fallback
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_DIR, use_fast=False)

model = AutoModelForSequenceClassification.from_pretrained(BASELINE_DIR)

enc_test, label_names, id2label, label2id = ensure_test_enc(tokenizer)

# ===== 평가 =====
args = TrainingArguments(
    output_dir="./tmp_eval_baseline",
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

print("== Baseline Test Metrics ==")
metrics = trainer.evaluate(enc_test)
print(metrics)

# ===== 샘플 예측 (정답 라벨과 확률만 출력) =====
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,   # GPU: 0, CPU: -1
    top_k=None       # 최상위 1개만 받기
)

samples = [
    "삼성전자, 반도체 업황 개선 기대감에 주가 급등",
    "정부, 부동산 정책 추가 발표… 전월세 시장 안정화 목표",
    "프로야구 한국시리즈 1차전, 연장 끝에 LG 승리",
    "구글, 차세대 AI 모델 공개… 개발자 행사서 발표",
]

print("\n== Baseline Predictions ==")
for s in samples:
    out = clf(s)
    # pipeline 반환 형식 방어적 처리:
    # - 보통 [{'label': '...', 'score': ...}]
    # - top_k 사용/환경에 따라 [[{...}, {...}, ...]]가 올 수도 있음
    if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
        pred = out[0]
    elif isinstance(out, list) and len(out) > 0 and isinstance(out[0], list):
        pred = out[0][0]
    else:
        # 혹시 다른 포맷이면 그대로 출력하고 다음으로
        print("- 예측 형식을 해석할 수 없어 raw 출력:", out)
        continue

    label = pred.get("label", "N/A")
    score = float(pred.get("score", 0.0)) * 100.0
    print(f"- {label} ({score:.1f}%)")
