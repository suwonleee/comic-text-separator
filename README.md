# comic-text-separator

만화/콘티 이미지에서 텍스트를 자동으로 검출하고, 레이어가 분리된 PSD 파일 또는 JSON 파일로 출력하는 CLI 도구.

---

## 기능

- 이미지 내 텍스트 자동 검출 (말풍선, 배경 텍스트 포함)
- OCR 텍스트 인식
- 텍스트 제거 (인페인팅) — 원본은 보존하고 텍스트만 제거
- 한국어 띄어쓰기 자동 교정
- PSD 출력 시 3단 레이어 구조:
  - 최하단: 원본 이미지
  - 중간: 텍스트 제거된 이미지
  - 최상단: 텍스트 레이어 그룹 (위치 보존)
- JSON 출력 시 텍스트 좌표, 폰트 크기, 색상, 방향 등 메타데이터 포함
- JSX 스크립트 자동 생성 — PSD 출력 시 Photopea/Photoshop에서 편집 가능한 텍스트 레이어 생성용 스크립트 동시 출력

---

## 사전 준비

- **Python 3.10 이상**
- **Git**

### Python 설치

- macOS: https://www.python.org/downloads/macos/ 에서 3.10 이상 버전 다운로드 후 설치
- Windows: https://www.python.org/downloads/windows/ 에서 3.10 이상 버전 다운로드 후 설치
  - 설치 시 **"Add Python to PATH" 체크박스를 반드시 체크**

### Git 설치

- macOS: 터미널에서 `git --version` 입력 시 자동 설치 안내가 뜸
- Windows: https://git-scm.com/download/win 에서 다운로드 후 설치 (기본 설정 유지)

---

## 설치

### 터미널 열기

- **macOS**: Spotlight(⌘ + Space) → "터미널" 검색 → 실행
- **Windows**: 시작 메뉴 → "cmd" 검색 → "명령 프롬프트" 실행

> 이 문서의 모든 명령어는 터미널(또는 명령 프롬프트) 에 직접 입력하는 것.

### 1단계: 프로젝트 다운로드

원하는 폴더로 이동한 뒤 아래 명령어 입력.

```bash
git clone https://github.com/suwonleee/comic-text-separator.git
cd comic-text-separator
```

- `git clone` — GitHub에서 프로젝트 전체를 내 컴퓨터로 복사하는 명령
- `cd` — 해당 폴더로 이동하는 명령

### 2단계: 의존성 설치

```bash
pip install -r requirements.txt
```

- `pip install` — 프로젝트 실행에 필요한 라이브러리를 자동으로 설치하는 명령
- `requirements.txt` — 필요한 라이브러리 목록 파일

> `pip`이 안 되면 `pip3 install -r requirements.txt`으로 시도.

> 첫 실행 시 ML 모델 파일 자동 다운로드 (약 500MB). 이후 재실행 시에는 다운로드 없이 즉시 실행.

---

## 사용법

### 기본 사용

1. `comic-text-separator/input/` 폴더에 변환할 이미지 파일 복사
2. 터미널에서 아래 명령어 입력:

```bash
python main.py
```

3. `output/` 폴더에 PSD 파일 + JSX 파일 생성 확인
4. [Photopea](https://www.photopea.com) 에서 PSD 파일을 열어 편집

- `python main.py` — 프로그램을 실행하는 명령
- `input/` — 변환할 원본 이미지를 넣는 폴더
- `output/` — 결과물이 저장되는 폴더

> `python`이 안 되면 `python3 main.py`로 시도.

### 편집 가능한 텍스트 레이어 적용 (JSX)

PSD 출력 시 같은 이름의 `.jsx` 파일이 함께 생성됨. 이 스크립트를 Photopea에서 실행하면 **편집 가능한 텍스트 레이어**가 생성됨.

1. [Photopea](https://www.photopea.com) 에서 출력된 PSD 파일 열기
2. 메뉴: **파일 > 스크립트...** 클릭
3. 스크립트 창의 텍스트 영역에 JSX 파일 내용 전체 복사 후 붙여넣기
4. **Run** 버튼 클릭
5. "Text Layers (Editable)" 그룹에 편집 가능한 텍스트 레이어 생성 확인

- JSX 파일 내 `FONT_NAME` 변수를 수정하여 원하는 폰트 지정 가능 (기본값: NanumGothic)
- JSX 실행 시 기존 래스터 텍스트 그룹("Text Layers")은 자동 삭제되고, 편집 가능한 텍스트 그룹으로 대체됨
- Photoshop에서도 동일하게 사용 가능 (파일 > 스크립트 > 찾아보기 → JSX 파일 선택)

### 전체 기능 사용 (텍스트 제거 + 띄어쓰기 교정 + PSD/JSON 동시 출력)

```bash
python main.py --format both --inpaint --correct-spacing
```

- `--format both` — PSD와 JSON 동시 출력 옵션
- `--inpaint` — 텍스트 영역을 배경으로 채워 제거하는 옵션
- `--correct-spacing` — 한국어 띄어쓰기 자동 교정 옵션

### JSON만 출력

```bash
python main.py --format json --correct-spacing
```

### 특정 이미지 또는 폴더 지정

```bash
python main.py -i 이미지경로.jpg -o 출력폴더/
```

- `-i` — 입력 경로 지정 (파일 하나 또는 폴더 전체)
- `-o` — 출력 폴더 지정

---

## 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-i`, `--input` | `input/` | 입력 이미지 또는 폴더 경로 |
| `-o`, `--output` | `output/` | 출력 폴더 경로 |
| `-f`, `--format` | `psd` | 출력 형식 (`psd`, `json`, `both`) |
| `--inpaint` | 꺼짐 | 텍스트 제거 (인페인팅) 활성화 |
| `--correct-spacing` | 꺼짐 | 한국어 띄어쓰기 교정 활성화 |
| `--detection-size` | `2048` | 텍스트 검출 이미지 크기 |
| `--text-threshold` | `0.5` | 텍스트 검출 임계값 |
| `--box-threshold` | `0.7` | 바운딩 박스 임계값 |
| `--inpainting-size` | `2048` | 인페인팅 이미지 크기 |
| `--use-gpu` | 꺼짐 | GPU 사용 (CUDA 또는 Apple Silicon MPS 자동 감지) |
| `-v`, `--verbose` | 꺼짐 | 상세 로그 출력 |

---

## 출력 파일

### PSD (Photopea/Photoshop용)

레이어 구조:

```
[Text Layers]               ← 텍스트 레이어 그룹 (최상단)
  ├── Text 1: 대사내용...
  ├── Text 2: 대사내용...
  └── ...
[Inpainted (Text Removed)]  ← 텍스트 제거된 이미지 (--inpaint 시)
[Original Image]             ← 원본 이미지 (최하단)
```

- 각 텍스트 레이어는 원본 위치에 배치됨
- `--correct-spacing` 사용 시 교정된 텍스트가 렌더링됨
- `--inpaint` 미사용 시 중간 레이어(Inpainted) 없이 2단 구조로 출력됨

### JSX (편집 가능한 텍스트 레이어용)

PSD 출력 시 자동 생성되는 ExtendScript 파일. Photopea 또는 Photoshop에서 실행하면 편집 가능한 텍스트 레이어 생성.

- 각 텍스트 영역의 위치, 크기, 색상, 폰트 크기를 원본 그대로 반영
- POINTTEXT (포인트 텍스트) 방식으로 생성 — 바운딩 박스 없이 텍스트 전체 표시
- 실행 시 래스터 텍스트 그룹("Text Layers")을 삭제하고 편집 가능한 그룹("Text Layers (Editable)")으로 대체
- `--correct-spacing` 사용 시 교정된 텍스트가 적용됨

### 인페인팅 이미지 (PNG)

`--inpaint` 사용 시 텍스트가 제거된 이미지가 `_inpainted.png` 파일로 별도 저장됨. PSD 내 Inpainted 레이어와 동일한 내용이며, 별도 활용 가능.

### JSON

```json
{
  "image_path": "/path/to/image.jpg",
  "image_width": 1280,
  "image_height": 20000,
  "regions": [
    {
      "text": "OCR 원본 텍스트",
      "text_corrected": "띄어쓰기 교정된 텍스트",
      "x": 100,
      "y": 200,
      "width": 300,
      "height": 80,
      "font_size": 40,
      "fg_color": [0, 0, 0],
      "direction": "h"
    }
  ]
}
```

---

## 프로젝트 구조

```
comic-text-separator/
├── main.py                  ← 진입점 (의존성 조립)
├── domain/
│   ├── models.py            ← 데이터 모델
│   └── ports.py             ← 인터페이스 정의
├── use_cases/
│   └── extract_text.py      ← 추출 파이프라인
├── adapters/
│   ├── detector.py          ← 텍스트 검출
│   ├── ocr_engine.py        ← OCR
│   ├── inpainter.py         ← 텍스트 제거
│   ├── textline_merger.py   ← 텍스트라인 병합
│   ├── textblock_mapper.py  ← 데이터 변환
│   └── kss_spacing.py       ← 띄어쓰기 교정
├── engine/                  ← ML 엔진 (텍스트 검출, OCR, 인페인팅)
│   ├── detection.py         ← DBNet 텍스트 검출
│   ├── recognition.py       ← OCR 인식
│   ├── inpainting.py        ← AOT 인페인팅
│   ├── merger.py            ← 텍스트라인 병합
│   ├── model_manager.py     ← 모델 다운로드/로드 관리
│   ├── types.py             ← TextLine, TextRegion
│   ├── geometry.py          ← 기하 연산
│   ├── image_utils.py       ← 이미지 전처리
│   └── nets/                ← 신경망 모듈
├── infrastructure/
│   ├── json_exporter.py     ← JSON 출력
│   ├── jsx_exporter.py      ← JSX 스크립트 출력
│   └── psd_exporter.py      ← PSD 출력
├── input/                   ← 입력 이미지 (gitignore)
├── output/                  ← 출력 결과 (gitignore)
├── requirements.txt
└── LICENSE
```

### 아키텍처

Clean Architecture 기반 5계층 구조. 의존성 방향은 안쪽(domain)으로만 향함.

```
main.py → adapters / infrastructure → use_cases → domain
                ↓
             engine
```

- **domain** — 순수 데이터 모델(`TextRegion`, `ExtractionResult`)과 포트(Protocol) 정의. 외부 의존성 없음
- **use_cases** — 추출 파이프라인 오케스트레이션. 포트 인터페이스에만 의존
- **adapters** — 포트 구현체. `engine/` 모듈을 래핑하여 domain 모델로 변환
- **infrastructure** — PSD/JSON/JSX 출력 담당. domain 모델만 참조
- **engine** — ML 추론 엔진. 나머지 계층과 독립적으로 동작 가능
- **main.py** — Composition Root. 모든 의존성을 여기서 조립 후 주입 (Constructor Injection)

포트는 Python `Protocol`(구조적 서브타이핑) 기반. ABC 상속 없이 메서드 시그니처 일치만으로 구현체 인정.

### ML 파이프라인

이미지 입력 → 결과 출력까지의 처리 흐름:

1. **텍스트 검출** — DBNet 기반 세그멘테이션. 이미지에서 텍스트 영역의 사각형 좌표 추출
2. **텍스트라인 병합** — 인접한 텍스트라인을 방향(가로/세로)과 거리 기준으로 하나의 텍스트 블록으로 병합
3. **OCR 인식** — 48px AR(Autoregressive) Transformer. 각 텍스트 블록에서 문자열 추출
4. **인페인팅** (선택) — AOT-GAN 기반. 텍스트 영역을 주변 배경으로 자연스럽게 채움
5. **띄어쓰기 교정** (선택) — py3langid로 언어 감지 후, 한국어 텍스트에만 kss 라이브러리로 맞춤법 기반 띄어쓰기 적용

### 모델 관리

- 첫 실행 시 GitHub Releases에서 자동 다운로드 → `models/` 폴더에 캐시
- SHA-256 해시 검증으로 파일 무결성 확인
- 총 용량 약 500MB (검출 294MB, OCR, 인페인팅 모델 포함)
- 재실행 시 캐시된 모델 즉시 로드

### GPU 지원

- `--use-gpu` 옵션으로 활성화
- NVIDIA GPU — CUDA 자동 감지
- Apple Silicon (M1/M2/M3) — MPS 백엔드 자동 감지
- GPU 미감지 시 CPU 자동 폴백

---

## 감사

[manga-image-translator](https://github.com/zyddnys/manga-image-translator), [psd-tools](https://github.com/psd-tools/psd-tools)에서 영감을 받았습니다.

## 라이선스

GPL-3.0. 전문은 [LICENSE](LICENSE) 파일 참조.
