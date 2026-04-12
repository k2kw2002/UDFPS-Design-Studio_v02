# 배포 가이드

## Hugging Face Spaces (추천)

### 1. HF 계정 생성
https://huggingface.co/join (무료)

### 2. Space 생성
https://huggingface.co/new-space
- Space name: `udfps-design-studio`
- SDK: **Docker**
- Visibility: Public (또는 Private)

### 3. 코드 업로드
```bash
# HF CLI 설치
pip install huggingface_hub

# 로그인
huggingface-cli login

# Space에 push
cd AI_PINN_PROJECT_v02
git init
git add -A
git commit -m "UDFPS BM Design Studio"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/udfps-design-studio
git push space main
```

### 4. 접속
https://huggingface.co/spaces/YOUR_USERNAME/udfps-design-studio
→ 자동 빌드 → 공유 URL 생성

### 5. GPU 활성화 (선택)
Space Settings → Hardware → T4 GPU (무료)
→ PINN 학습도 클라우드에서 가능

---

## 로컬 실행
```bash
python start.py           # http://localhost:8000
python start.py --share   # ngrok 외부 공유
```

## Docker
```bash
docker build -t design-studio .
docker run -p 8000:8000 design-studio
```
