summary = {
    "settings": {
        "model": MODEL_NAME,
        "subset_size_each_split": SUBSET_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "lr": LR,
    },
    "labels": label_names,
    "Fine-tuning": {
        "val_accuracy": float(baseline_val["eval_accuracy"]),
        "test_accuracy": float(baseline_test["eval_accuracy"]),
        "train_time_sec": int(baseline_train_seconds),
        "gpu_peak_bytes": int(baseline_gpu_peak),
        "gpu_peak_human": human_size(baseline_gpu_peak),
        "model_dir": BASELINE_DIR,
        "model_size_bytes": int(baseline_size),
        "model_size_human": human_size(baseline_size),
    },
    "lora": {
        "val_accuracy": float(lora_val["eval_accuracy"]),
        "test_accuracy": float(lora_test["eval_accuracy"]),
        "train_time_sec": int(lora_train_seconds),
        "gpu_peak_bytes": int(lora_gpu_peak),
        "gpu_peak_human": human_size(lora_gpu_peak),
        "model_dir": LORA_DIR,
        "model_size_bytes": int(lora_size),
        "model_size_human": human_size(lora_size),
    }
}

print(json.dumps(summary, ensure_ascii=False, indent=2))
