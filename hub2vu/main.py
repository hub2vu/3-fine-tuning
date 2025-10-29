import os
import subprocess

# 현재 스크립트가 있는 디렉터리 기준으로 실행
base_dir = os.path.dirname(os.path.abspath(__file__))

# 파일 목록을 가져오되, py 파일만 필터링
files = [f for f in os.listdir(base_dir) if f.endswith(".py") and f[0].isdigit()]

# 숫자 순서대로 정렬
files.sort(key=lambda x: int(''.join(ch for ch in x if ch.isdigit())))

print("실행 순서:")
for f in files:
    print(f" - {f}")

# 각 파일을 순서대로 실행
for f in files:
    print(f"\n▶ 실행 중: {f}")
    subprocess.run(["python", os.path.join(base_dir, f)], check=True)
