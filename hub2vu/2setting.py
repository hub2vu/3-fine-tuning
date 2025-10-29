import os, time, json, math, shutil, subprocess, gc, sys
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, set_seed)
from sklearn.metrics import accuracy_score
from peft import LoraConfig, get_peft_model, TaskType
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))


# ===== 사용자 조정 파트 =====
MODEL_NAME = "klue/roberta-base"   # 한국어에 강한 베이스 모델
MAX_LENGTH = 128                    # 제목 길이가 짧아 128로 충분
BATCH_SIZE = 32                     # GPU 여유에 따라 16~64 조정
LR = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# 전체(=None) 또는 임의의 N개만 사용
SUBSET_SIZE = None   # 예: 500으로 두면 각 split에서 랜덤 500개만 사용

OUTPUT_DIR_BASE = "/content/ynat_runs"
BASELINE_DIR = f"{OUTPUT_DIR_BASE}/baseline_roberta"
LORA_DIR = f"{OUTPUT_DIR_BASE}/lora_roberta"

os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
set_seed(SEED)