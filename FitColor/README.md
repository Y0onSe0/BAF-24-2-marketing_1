# 🎨 FitColor — CV 기반 패션 색상 추천 웹앱

> **패션에 무지한 남성**을 위한 색상 코디 도우미  
> 상의 사진 하나만 찍으면 어울리는 하의 색상을 추천하고, 바로 쇼핑으로 연결합니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **타겟** | 패션 코디에 어려움을 겪는 남성 |
| **데이터** | 룩핀 코디 이미지 크롤링 (상하의 색상 쌍 867개) |
| **모델** | YOLOv5s (상/하의 감지, mAP50 96.4%) |
| **추천** | KMeans(k=200) Palette + KNN 기반 색상 추천 |
| **배포** | Streamlit 웹앱 |

---

## 🔄 파이프라인

```
사진 입력
  ↓ YOLOv5  — 상의 / 하의 영역 감지
  ↓ KMeans(k=1)  — 영역별 대표 색상 추출
  ↓ find_nearest_cluster  — Palette(k=200)에서 유사 클러스터 탐색
  ↓ 다양성 확보(k=5)  — 추천 하의 색상 5가지 선정
  OUTPUT: 색상명 + 무신사 쇼핑 링크
```

---

## 🗂️ 파일 구조

```
├── app.py                  # Streamlit 메인 앱
├── build_palette.py        # Palette CSV 생성 스크립트 (최초 1회 실행)
├── bottom_rgb_list.csv     # KMeans k=200 클러스터 결과 (Palette 데이터셋)
├── 최종2411022.csv          # 원본 색상 쌍 데이터 (866쌍)
├── requirements.txt
└── 최종코드/
    ├── 1. yolov5학습.ipynb
    ├── 2. 색상추출.ipynb
    ├── 3. 색상 클러스터링.ipynb
    └── yolov5/
        └── runs/train/best_yolov5s_result2/weights/best.pt  ← 학습된 모델
```

---

## 🚀 실행 방법

```bash
pip install -r requirements.txt

# Palette CSV 없으면 먼저 생성
python build_palette.py

# 앱 실행
streamlit run app.py
```

---

## 🛠️ 기술 스택

- **Object Detection**: YOLOv5s (Ultralytics)
- **Color Extraction**: KMeans Clustering (scikit-learn)
- **Recommendation**: KNN + Palette Clustering
- **Web**: Streamlit
- **Data**: 룩핀 크롤링 (Selenium, BeautifulSoup)

---

## ⚠️ 한계 및 개선 방향

| 한계 | 개선 방향 |
|------|-----------|
| 학습 데이터 510장, normcore 편향 | 다양한 스타일 데이터 추가 수집 |
| 단일 대표 색상 추출 (패턴 약함) | 컬러 히스토그램 기반 유사도로 전환 |
| 룩핀 코디 사진에 최적화 | 실사 사진 대응 위한 데이터 증강 |

---

## 👥 팀

BAF 24-2 마케팅 세션
