# Windows 빌드 환경 설정

gsplat 1.x를 Windows (한국어 로케일, MSVC)에서 JIT 컴파일할 때 세 가지 문제가 발생한다.

---

## 문제 1: UnicodeDecodeError (cp949)

**증상:**
```
UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2 in position 4617
```

**원인:** PyTorch의 JIT 버전 관리 코드가 gsplat CUDA 소스 파일을 텍스트 모드로 읽을 때 시스템 기본 인코딩(cp949)을 사용.

**수정:**
```python
import torch.utils._cpp_extension_versioner as _v

def _patched_hash(hash_value, source_files):
    for filepath in source_files:
        with open(filepath, "rb") as f:          # binary 모드로 읽기
            hash_value = _v.update_hash(
                hash_value,
                f.read().decode("utf-8", errors="replace"),
            )
    return hash_value

_v.hash_source_files = _patched_hash
```

---

## 문제 2: cl.exe not found

**증상:**
```
subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1
```

**원인:** MSVC `cl.exe`가 PATH에 없음.

**수정:**
```python
# Visual Studio 설치 경로에서 cl.exe 탐색하여 PATH 추가
vs_roots = [
    r"C:\Program Files\Microsoft Visual Studio\2022",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022",
]
for vs_root in vs_roots:
    pattern = os.path.join(vs_root, "**", "Hostx64", "x64", "cl.exe")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        cl_dir = os.path.dirname(matches[0])
        os.environ["PATH"] = cl_dir + os.pathsep + os.environ["PATH"]
        break
```

---

## 문제 3: D8021 — `/Wno-attributes` invalid (핵심)

**증상:**
```
cl : Command line error D8021 : invalid numeric argument '/Wno-attributes'
ninja: build stopped: subcommand failed.
```

**원인:** gsplat 1.x `_backend.py`의 177번 줄:
```python
extra_cflags = [opt_level, "-Wno-attributes"]   # GCC 전용 플래그
```
이 플래그가 MSVC에 전달되어 오류 발생.

**수정:** gsplat이 `from torch.utils.cpp_extension import _jit_compile`으로 로컬 바인딩하므로, **gsplat import 전에** `_jit_compile`을 패치해야 함:

```python
import torch.utils.cpp_extension as _ext

_GCC_ONLY = frozenset([
    "-O0", "-O1", "-O2", "-O3", "-Og", "-Os",
    "-Wno-attributes", "-Wno-unused-variable",
    "-fPIC", "-fvisibility=hidden",
])
_orig = _ext._jit_compile

def _patched(*args, **kwargs):
    args = list(args)
    # extra_cflags는 3번째 positional arg (index 2)
    if len(args) > 2 and args[2] and os.name == "nt":
        cleaned = [f for f in args[2] if f not in _GCC_ONLY]
        if not any(f.startswith("/O") for f in cleaned):
            cleaned.append("/O2")   # MSVC 최적화 플래그
        args[2] = cleaned
    return _orig(*args, **kwargs)

_ext._jit_compile = _patched
```

> **핵심:** `torch.utils.cpp_extension.load`가 아닌 `_jit_compile`을 패치해야 한다. gsplat이 `_jit_compile`을 직접 import하기 때문.

---

## 통합: build_env.py

```python
def setup_windows_cuda_env():
    if os.name != "nt":
        return
    _fix_gsplat_msvc_flags()   # 문제 3 수정
    _fix_versioner_encoding()  # 문제 1 수정
    _setup_msvc()              # 문제 2 수정
    _setup_cccl()              # CUDA CCCL include path 추가
```

모든 스크립트 상단에서 gsplat import 전에 호출:

```python
import sys
sys.path.insert(0, ".")
from src.utils.build_env import setup_windows_cuda_env
setup_windows_cuda_env()   # ← gsplat import 전에 반드시 호출

from gsplat import rasterization   # 이제 정상 동작
```

---

## 환경 정보

| 항목 | 버전 |
|---|---|
| Python | 3.8 |
| PyTorch | 2.4.1+cu121 |
| CUDA | 12.1 |
| gsplat | ≥ 1.0.0 |
| MSVC | 19.38.33145 (VS2022) |
| OS | Windows 11 (한국어) |
