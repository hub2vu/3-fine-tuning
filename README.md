# 3-finetuning
## fine-tuning 모델 학습 실습과 과제 제출을 위한 Repository입니다 

### 일반 fine-tuning과 lora 모델 활용 학습의 성능 차이 비교
사용 모델: --

|항목|일반 파인튜닝|LoRA 파인튜닝|
|---:|---:|---:|
|Validation Accuracy|-0.87-|-0.85-|
|Training Time|-515-|-446-|
|GPU 메모리 사용량|-2.40 GB-|-4.23 GB-|
|모델 저장 용량|-4.13 GB-|-78.59 MB-|
|etc|

### 예시 

```bash
!git clone https:/[토큰값]@github.com/HateSlop/3-finetuning.git  # 클론
!cd 3-finetuning # 프로젝트 루트로 이동
!git checkout -b '본인 브랜치명' # 브랜치 생성 (본인의 브랜치, 폴더 등 생성)
!mkdir '본인 폴더명' # 개인 폴더 만들기
!cd '본인 폴더명' # 개인 폴더로 이동
# 작업을 진행해주세요
!git add . # 작업 후 add
!git commit -m "[feat] ~~" # 커밋
!git push origin '본인 브랜치명' # 오리진에 푸시
```

### 폴더구조
```bash
├── fine-tuning.ipynb
├── terminal.ipynb
├── lora.ipynb
├── README.md
```

### 커밋 컨벤션

feat: 새로운 기능 추가  
fix: 버그 수정  
docs: 문서 수정  
style: 코드 포맷팅, 세미콜론 누락, 코드 변경이 없는 경우  
refactor: 코드 리팩토링  
test: 테스트 코드, 리팩토링 테스트 코드 추가  
chore: 빌드 업무 수정, 패키지 매니저 수정, production code와 무관한 부분들 (.gitignore, build.gradle 같은)  
comment: 주석 추가 및 변경  
remove: 파일, 폴더 삭제  
rename: 파일, 폴더명 수정
