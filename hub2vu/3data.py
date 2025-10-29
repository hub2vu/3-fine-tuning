# 0) 로드
raw = load_dataset("klue", "ynat")
print("Available splits:", list(raw.keys()))  # ['train', 'validation'] 만 있을 것

label_names = raw["train"].features["label"].names
num_labels  = len(label_names)
label2id    = {n:i for i,n in enumerate(label_names)}
id2label    = {i:n for n,i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def preprocess(batch):
    enc = tokenizer(batch["title"], truncation=True, max_length=MAX_LENGTH)
    enc["labels"] = batch["label"]
    return enc

def maybe_subsample(ds, n=None, seed=SEED):
    if n is None or n >= len(ds): 
        return ds
    return ds.shuffle(seed=seed).select(range(n))

# 1) 기본 split 구성
# - validation: 제공된 검증 세트 그대로 사용
# - test: train에서 10%를 라벨 분포 유지(stratify)하며 분리
train_split = raw["train"].train_test_split(
    test_size=0.1, seed=SEED, stratify_by_column="label"
)
ds_train = train_split["train"]
ds_test  = train_split["test"]
ds_val   = raw["validation"]

# (옵션) 샘플 수 축소
ds_train = maybe_subsample(ds_train, SUBSET_SIZE)
ds_val   = maybe_subsample(ds_val,   SUBSET_SIZE)
ds_test  = maybe_subsample(ds_test,  SUBSET_SIZE)

# 2) 전처리
enc_train = ds_train.map(preprocess, batched=True, remove_columns=ds_train.column_names)
enc_val   = ds_val.map(preprocess,   batched=True, remove_columns=ds_val.column_names)
enc_test  = ds_test.map(preprocess,  batched=True, remove_columns=ds_test.column_names)

for ds in (enc_train, enc_val, enc_test):
    ds.set_format(type="torch")

print(enc_train, enc_val, enc_test, sep="\n")
print("Labels:", label_names)

