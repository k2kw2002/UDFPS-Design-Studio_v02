# LightTools 연동 가이드 (회사 설정용)

## 사전 준비 (회사 PC에서)

### 1. Python 패키지 설치
```bash
pip install pywin32
```

### 2. LightTools COM 서버 등록 확인
LightTools가 설치되어 있으면 COM 서버가 자동 등록됩니다.
확인 방법:
```python
python -c "import win32com.client; lt = win32com.client.Dispatch('LightTools.LTAPI'); print('OK')"
```
- `OK` 출력 → COM 연결 성공
- 에러 → LightTools 버전에 따라 COM 이름이 다를 수 있음 (아래 트러블슈팅 참고)

---

## 단계별 연동

### Step 1: LightTools COM 이름 확인

**이게 가장 중요합니다.** 현재 코드의 `"LightTools.LTAPI"`가 맞는지 확인해야 합니다.

```python
# 방법 1: 직접 시도
import win32com.client
lt = win32com.client.Dispatch("LightTools.LTAPI")

# 방법 2: LT 버전에 따른 대안
# lt = win32com.client.Dispatch("LightTools.LTAPI2")
# lt = win32com.client.Dispatch("LightTools.Application")
# lt = win32com.client.Dispatch("LTAPI.Application")
```

**확인 방법 (Windows)**:
1. `Win+R` → `dcomcnfg` 입력 → Enter
2. Component Services → Computers → My Computer → DCOM Config
3. "LightTools" 검색 → 정확한 COM 이름 확인

또는:
```python
# 레지스트리에서 COM 이름 검색
import winreg
key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "")
i = 0
while True:
    try:
        name = winreg.EnumKey(key, i)
        if "light" in name.lower():
            print(name)
        i += 1
    except WindowsError:
        break
```

### Step 2: LT 모델 파일 열기

LightTools에 이미 만들어진 UDFPS BM 모델(.ltp)이 있어야 합니다.

```python
import win32com.client

lt = win32com.client.Dispatch("LightTools.LTAPI")  # Step 1에서 확인한 이름
lt.Open(r"C:\LT_Models\UDFPS_BM_7pitch.ltp")      # 모델 경로
```

### Step 3: 모델 내부 객체 이름 확인

**이게 두 번째로 중요합니다.** 코드의 `"BM1.Aperture[{i}].Center"` 같은 경로가 실제 LT 모델의 객체 이름과 맞아야 합니다.

```python
# LT 모델의 객체 트리 탐색 (LT 버전마다 API 다름)
# 방법 1: LT GUI에서 직접 확인
#   → 모델 트리에서 BM1 레이어 이름, 아퍼처 이름 확인

# 방법 2: API로 탐색 (가능한 경우)
# LT 8.x 이상:
print(lt.GetObjectList())     # 최상위 객체 목록
print(lt.GetChildObjects("BM1"))  # BM1 하위 객체

# 또는 LT Python API 문서의 예제 참고
```

**확인해야 할 이름들:**

| 코드에서 사용 중 | 실제 LT 모델에서 확인 필요 |
|-----------------|-------------------------|
| `BM1.Aperture[{i}].Center` | BM1 레이어의 아퍼처 중심 좌표 파라미터명 |
| `BM1.Aperture[{i}].Width` | BM1 아퍼처 폭 파라미터명 |
| `BM2.Aperture[{i}].Center` | BM2 레이어의 아퍼처 중심 좌표 파라미터명 |
| `BM2.Aperture[{i}].Width` | BM2 아퍼처 폭 파라미터명 |
| `OPD[{i}]` | OPD 리시버 이름 |
| `RunRayTrace(NumRays=...)` | Ray trace 실행 메소드명 |
| `GetReceiverPower(...)` | 리시버 파워 조회 메소드명 |

### Step 4: 테스트 실행

모든 이름이 확인되면 단일 실행 테스트:

```python
from backend.data.lighttools_runner import LightToolsRunner
from backend.api.schemas import BMDesignParams

# 연결
lt = LightToolsRunner(lt_model_path=r"C:\LT_Models\UDFPS_BM_7pitch.ltp")

# 테스트: baseline 설계 (d1=d2=0, w1=w2=10)
params = BMDesignParams(delta_bm1=0, delta_bm2=0, w1=10, w2=10)
psf = lt.run_single(params)

if psf is not None:
    print(f"PSF7 = {psf}")
    print(f"Peak = {psf.max():.4f}, Contrast = {(psf.max()-psf.min())/(psf.max()+psf.min()):.2%}")
else:
    print("실패 — 에러 로그 확인")
```

### Step 5: 배치 수집 + CSV 저장

테스트 성공 후 학습 데이터 수집:

```python
from backend.data.lhs_sampler import generate_lhs_samples

# 20개 LHS 샘플 생성
samples = generate_lhs_samples(n=20)

# LT 배치 실행
results = lt.run_batch(samples)
print(f"{len(results)}개 성공")

# CSV 저장
lt.save_to_csv(results, "lt_data.csv")
# → backend/data/lt_results/lt_data.csv 에 저장됨
```

---

## 트러블슈팅

### "LightTools.LTAPI" 연결 실패

```
pywintypes.com_error: (-2147221005, 'Invalid class string', ...)
```

**원인**: COM 이름이 다름
**해결**:
1. `dcomcnfg`에서 정확한 이름 확인
2. 또는 LT 설치 폴더에서 `.tlb` 파일 찾기:
   ```
   dir "C:\Program Files\Synopsys\LightTools\*.tlb" /s
   ```
3. LT Python API 문서 (보통 LT 설치 폴더/Documentation/PythonAPI)

### SetParameter 경로 오류

```
lt_api.SetParameter("BM1.Aperture[0].Center", 0)
→ "Parameter not found"
```

**원인**: 모델 내 객체 이름이 다름
**해결**:
1. LT GUI에서 모델 트리 열기
2. BM1 레이어 → 아퍼처 → 속성 이름 직접 확인
3. `SetParameter`가 아니라 다른 메소드일 수 있음:
   ```python
   # 가능한 대안들
   lt.SetProperty("BM1", "ApertureCenter", value)
   lt.SetSurfaceParameter("BM1", "center_x", value)
   lt.DbSet("BM1.Aperture[0].Center", value)
   ```

### RunRayTrace 메소드 없음

**LT 버전에 따라**:
```python
# 가능한 대안들
lt.RunRayTrace()               # 파라미터 없이
lt.Run()                       # 간단한 버전
lt.Simulate(nRays=1000000)     # 다른 메소드명
lt.Execute("RayTrace")         # 명령 문자열
```

### GetReceiverPower 없음

```python
# 가능한 대안들
power = lt.GetDetectorData("OPD[0]")
power = lt.GetReceiverData("OPD[0]", "TotalPower")
power = lt.DbGet("OPD[0].Power")
```

---

## LT 모델이 아직 없는 경우

LT에서 새로 만들어야 할 구조:

```
모델 구조:
├── 광원: 면광원 (504um × 504um), 파장 520nm
│         Lambertian 분포, 각도 ±41.1°
│
├── Cover Glass: 550um, n=1.52
│
├── BM1 레이어 (z=40um 위치):
│   ├── 7개 사각 아퍼처 (pitch=72um)
│   ├── 아퍼처 폭: 기본 10um (파라미터화)
│   └── 아퍼처 오프셋: 기본 0um (파라미터화)
│
├── ILD: 20um, n≈1.5
│
├── BM2 레이어 (z=20um 위치):
│   ├── 7개 사각 아퍼처 (BM1과 동일 구조)
│   └── 별도 오프셋/폭 파라미터
│
└── OPD 면 (z=0um):
    └── 7개 리시버 (72um 간격, 중심에 배치)
```

---

## 확인된 후 코드 수정

LT에서 실제 API 이름을 확인하면, `lighttools_runner.py`를 수정합니다:

```python
# 예시: 실제 확인된 이름으로 교체
class LightToolsRunner:
    def __init__(self, lt_model_path):
        import win32com.client
        self.lt_api = win32com.client.Dispatch("실제COM이름")
        self.lt_api.Open(lt_model_path)

    def run_single(self, params):
        # BM1 설정 — 실제 파라미터 경로로 교체
        for i in range(7):
            x_center = (i - 3) * 72.0 + params.delta_bm1
            self.lt_api.실제메소드("실제경로", x_center)
            self.lt_api.실제메소드("실제경로", params.w1)

        # Ray trace — 실제 메소드명
        self.lt_api.실제RayTrace메소드()

        # 결과 수집 — 실제 메소드명
        psf7 = np.zeros(7)
        for i in range(7):
            psf7[i] = self.lt_api.실제결과조회("실제리시버이름")

        return psf7
```

---

## 내일 체크리스트

- [ ] `pip install pywin32` 설치
- [ ] COM 이름 확인 (`dcomcnfg` 또는 레지스트리 검색)
- [ ] LT 모델 파일 경로 확인 (.ltp)
- [ ] LT GUI에서 BM1/BM2 객체 이름 확인
- [ ] LT GUI에서 OPD 리시버 이름 확인
- [ ] `SetParameter` / `RunRayTrace` / `GetReceiverPower` 실제 API 메소드명 확인
- [ ] 단일 테스트 실행 (baseline d=0, w=10)
- [ ] 성공하면 → 20개 LHS 배치 수집
- [ ] CSV 저장 확인 (backend/data/lt_results/lt_data.csv)
