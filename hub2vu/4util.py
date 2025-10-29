from datetime import timedelta

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

def human_size(num_bytes: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    i = 0
    n = float(num_bytes)
    while n >= 1024 and i < len(units)-1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}"

def get_dir_size(path: str) -> int:
    total = 0
    p = Path(path)
    if not p.exists():
        return 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total

def gpu_mem_peak_bytes():
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_reserved()  # reserved가 보수적으로 기록됨

def reset_gpu_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def print_nvidia_smi():
    if not torch.cuda.is_available():
        print("No CUDA.")
        return
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,name",
                                       "--format=csv,noheader"], text=True)
        print(out)
    except Exception as e:
        print("nvidia-smi error:", e)