import streamlit as st
import torch
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import ast
import sys
import os
import base64
import io
import cv2 as _cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '최종코드', 'yolov5'))

MODEL_PATH = os.path.join(os.path.dirname(__file__), '최종코드', 'yolov5', 'runs', 'train', 'best_yolov5s_result2', 'weights', 'best.pt')
PALETTE_PATH = os.path.join(os.path.dirname(__file__), 'bottom_rgb_list.csv')

@st.cache_resource
def load_model():
    model = torch.hub.load(
        os.path.join(os.path.dirname(__file__), '최종코드', 'yolov5'),
        'custom', path=MODEL_PATH, source='local', verbose=False
    )
    model.conf = 0.3
    return model

@st.cache_data
def load_palette():
    """발표자료 Palette 데이터셋 로드 (KMeans k=200 클러스터 결과)"""
    df = pd.read_csv(PALETTE_PATH)
    df['Representative_Color'] = df['Representative_Color'].apply(ast.literal_eval)
    df['Bottom_RGB_List']      = df['Bottom_RGB_List'].apply(ast.literal_eval)
    return df

def extract_major_colors(image):
    """바운딩박스 이미지에서 주요 색상 추출 (KMeans k=1)"""
    pixels = np.array(image)[:, :, :3].reshape(-1, 3)
    km = KMeans(n_clusters=1, n_init=10, random_state=42)
    km.fit(pixels)
    return tuple(map(int, km.cluster_centers_[0]))

def find_nearest_cluster(top_rgb, palette_df):
    """상의 색상과 가장 유사한 Palette 클러스터 찾기 → 하의 색상 리스트 반환"""
    centers = np.array(palette_df['Representative_Color'].tolist())
    query   = np.array(top_rgb).reshape(1, -1)
    dists   = np.linalg.norm(centers - query, axis=1)
    nearest = np.argmin(dists)
    return palette_df.iloc[nearest]['Bottom_RGB_List']

def rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

# 색상 이름 매핑 (RGB → 한국어 색상명)
COLOR_NAMES = {
    '블랙':        (10,  10,  10),
    '다크 네이비': (20,  25,  50),
    '네이비':      (30,  50,  100),
    '차콜':        (55,  60,  65),
    '다크 그레이': (80,  82,  85),
    '그레이':      (130, 130, 130),
    '라이트 그레이':(190,190, 190),
    '아이보리':    (230, 220, 195),
    '화이트':      (240, 240, 240),
    '베이지':      (195, 175, 140),
    '카멜':        (175, 130, 75),
    '브라운':      (110, 75,  45),
    '다크 브라운': (65,  40,  25),
    '카키':        (90,  95,  55),
    '올리브':      (100, 105, 60),
    '다크 올리브': (65,  70,  40),
    '버건디':      (100, 30,  40),
    '와인':        (80,  20,  35),
    '블루':        (60,  100, 170),
    '라이트 블루': (150, 185, 220),
    '민트':        (140, 195, 185),
    '그린':        (80,  140, 100),
    '머스타드':    (195, 165, 55),
}

_color_arr   = np.array(list(COLOR_NAMES.values()))
_color_names = list(COLOR_NAMES.keys())

def get_color_name(rgb):
    dists = np.linalg.norm(_color_arr - np.array(rgb), axis=1)
    return _color_names[int(np.argmin(dists))]

def shopping_urls(color_name):
    q = f'{color_name} 바지'
    musinsa = f'https://www.musinsa.com/search/goods?q={q}'
    return musinsa

def swatch_card(rgb, size=72):
    """색상 카드: 스와치 + 색상명 + 쇼핑 링크"""
    hex_c = rgb_to_hex(rgb)
    name  = get_color_name(rgb)
    url   = shopping_urls(name)
    return f'''
<div style="display:inline-block;text-align:center;margin:6px;width:{size+16}px;">
  <div style="width:{size}px;height:{size}px;background:{hex_c};border-radius:12px;
              border:1px solid #e0e0e0;box-shadow:0 2px 8px rgba(0,0,0,0.1);margin:0 auto;"></div>
  <div style="font-size:12px;font-weight:600;color:#333;margin-top:6px;">{name}</div>
  <div style="font-size:10px;color:#aaa;margin-top:1px;">{hex_c}</div>
  <a href="{url}" target="_blank"
     style="display:inline-block;margin-top:6px;padding:3px 10px;background:#222;color:#fff;
            border-radius:20px;font-size:10px;text-decoration:none;font-weight:600;">
    무신사 검색 →
  </a>
</div>'''

def swatch_html(rgb, size=60, label=None):
    hex_c = rgb_to_hex(rgb)
    name  = get_color_name(rgb)
    lbl   = f'<div style="font-size:12px;font-weight:600;color:#333;margin-top:4px;">{name}</div><div style="font-size:10px;color:#aaa;">{hex_c}</div>'
    return f'<div style="display:inline-block;text-align:center;margin:4px;"><div style="width:{size}px;height:{size}px;background:{hex_c};border-radius:10px;border:1px solid #ddd;box-shadow:0 2px 6px rgba(0,0,0,0.12);"></div>{lbl}</div>'

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(page_title='FitColor', page_icon='🎨', layout='centered')

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #f8f9fb; }
    .block-container { padding-top: 2rem; max-width: 760px; }
    h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    .tag {
        display: inline-block;
        background: #f0f0f0;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        color: #555;
        margin-right: 6px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-size: 14px;
    }
    .stFileUploader > div { border-radius: 12px; }
    .warn-box {
        background: #fff8e1;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 14px;
        color: #555;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── 헤더 ────────────────────────────────────────────────────
st.markdown("# 🎨 FitColor")
st.markdown("<p style='color:#888;margin-top:-12px;'>상의 색상을 분석해 어울리는 하의 색상을 추천합니다</p>", unsafe_allow_html=True)
st.markdown("<span class='tag'>YOLOv5</span><span class='tag'>KMeans</span><span class='tag'>KNN</span>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── 이미지 업로드 ────────────────────────────────────────────
tab1, tab2 = st.tabs(['📁  파일 업로드 / 드래그앤드롭', '📷  카메라로 촬영'])

with tab1:
    uploaded = st.file_uploader('', type=['jpg', 'jpeg', 'png'], label_visibility='collapsed')

with tab2:
    cam = st.camera_input('')

image = None
if uploaded:
    image = Image.open(uploaded).convert('RGB')
elif cam:
    image = Image.open(cam).convert('RGB')

# ── 분석 ────────────────────────────────────────────────────
if image:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(image, use_container_width=True, caption='입력 이미지')
    with col_info:
        st.markdown("**분석 중...**")
        prog = st.progress(0)

    try:
        prog.progress(20)
        model = load_model()
        prog.progress(50)
        resized = image.resize((400, 500))
        results = model(resized)
        prog.progress(80)
        detections = results.xyxy[0].cpu().numpy()
        prog.progress(100)
        prog.empty()
    except Exception as e:
        st.error(f'모델 로드 실패: {e}')
        st.stop()

    top_rgb = None
    bottom_rgb_detected = None

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        crop = resized.crop((int(x1), int(y1), int(x2), int(y2)))
        color = extract_major_colors(crop)
        if int(cls) == 0:
            top_rgb = color
        elif int(cls) == 1:
            bottom_rgb_detected = color

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 결과 ──────────────────────────────────────────────────
    if top_rgb is None and bottom_rgb_detected is None:
        st.markdown("""
        <div class='warn-box'>
        ⚠️ 상의 또는 하의를 감지하지 못했습니다.<br>
        <b>전신이 나오는 코디 사진</b>을 사용하면 더 잘 됩니다.
        </div>
        """, unsafe_allow_html=True)
    else:
        # 감지된 색상
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 감지된 색상")
        det_cols = st.columns([1, 1, 2])
        with det_cols[0]:
            if top_rgb:
                st.markdown(swatch_html(top_rgb, size=70, label='상의'), unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#bbb;font-size:13px;padding:8px;'>상의 미감지</div>", unsafe_allow_html=True)
        with det_cols[1]:
            if bottom_rgb_detected:
                st.markdown(swatch_html(bottom_rgb_detected, size=70, label='하의'), unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#bbb;font-size:13px;padding:8px;'>하의 미감지</div>", unsafe_allow_html=True)
        with det_cols[2]:
            # 감지 박스 이미지
            det_img = np.array(resized).copy()
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                c = (76, 175, 80) if int(cls) == 0 else (255, 152, 0)
                lbl = f"{'TOP' if int(cls)==0 else 'BOTTOM'} {conf:.0%}"
                _cv2.rectangle(det_img, (x1, y1), (x2, y2), c, 2)
                _cv2.rectangle(det_img, (x1, y1-22), (x1+len(lbl)*9, y1), c, -1)
                _cv2.putText(det_img, lbl, (x1+2, y1-6), _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            st.image(det_img, use_container_width=True, caption='감지 결과')
        st.markdown("</div>", unsafe_allow_html=True)

        # 추천 색상
        if top_rgb:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("#### ✨ 추천 하의 색상")
            top_name = get_color_name(top_rgb)
            st.markdown(f"<p style='color:#888;font-size:13px;margin-top:-8px;'><b>{top_name}</b> 상의와 어울리는 하의 색상 — 클릭하면 바로 쇼핑으로</p>", unsafe_allow_html=True)
            palette_df  = load_palette()
            bottom_list = find_nearest_cluster(top_rgb, palette_df)
            bottom_arr  = np.array(bottom_list).astype(float)
            n = min(5, len(bottom_arr))
            km5 = KMeans(n_clusters=n, n_init=10, random_state=42)
            km5.fit(bottom_arr)
            recs = [tuple(map(int, c)) for c in km5.cluster_centers_]
            cards = "".join(swatch_card(r) for r in recs)
            st.markdown(f"<div style='display:flex;gap:4px;flex-wrap:wrap;margin-top:12px;'>{cards}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info('상의가 감지되지 않아 추천을 생성할 수 없습니다.')

# ── 푸터 ────────────────────────────────────────────────────
st.markdown("<br><hr style='border:none;border-top:1px solid #eee;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#bbb;font-size:12px;'>FitColor · YOLOv5 기반 패션 색상 추천 · BAF 24-2</p>", unsafe_allow_html=True)
