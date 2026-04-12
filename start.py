"""
start.py - Design Studio 원클릭 실행
=====================================
python start.py          → 로컬 서버 (http://localhost:8000)
python start.py --share  → 외부 공유 (ngrok URL 생성)
python start.py --port 9000  → 포트 변경
"""
import argparse, subprocess, sys, webbrowser, time

parser = argparse.ArgumentParser(description="UDFPS BM Design Studio")
parser.add_argument('--port', type=int, default=8000)
parser.add_argument('--share', action='store_true', help='ngrok으로 외부 공유')
args = parser.parse_args()

print("=" * 50)
print("  UDFPS BM Design Studio")
print("=" * 50)

# 모델 상태 확인
from pathlib import Path
models = {
    "Parametric PINN": Path("parametric_pinn_ckpt.pt").exists(),
    "FNO Surrogate": Path("fno_surrogate_ckpt.pt").exists(),
    "BoTorch Top-5": Path("botorch_top5.pt").exists(),
    "AR LUT": Path("backend/physics/ar_coating/data/ar_lut.npz").exists(),
}
print("\nModels:")
for name, ok in models.items():
    print(f"  {'[OK]' if ok else '[  ]'} {name}")

# 서버 시작
print(f"\nStarting server on port {args.port}...")
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.api.main:app",
     "--host", "0.0.0.0", "--port", str(args.port)],
)
time.sleep(2)

url = f"http://localhost:{args.port}"

if args.share:
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(args.port)
        url = str(public_url)
        print(f"\n  Public URL: {url}")
        print(f"  Share this URL with colleagues!")
    except Exception as e:
        print(f"\n  ngrok failed: {e}")
        print(f"  Install: pip install pyngrok")
        print(f"  Then: ngrok authtoken YOUR_TOKEN")

print(f"\n  Local URL: http://localhost:{args.port}")
print(f"\n  Press Ctrl+C to stop")

webbrowser.open(url)

try:
    server.wait()
except KeyboardInterrupt:
    print("\nShutting down...")
    server.terminate()
