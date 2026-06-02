"""
867쌍 CSV → KMeans(k=200) → bottom_rgb_list.csv 생성
발표자료 파이프라인 재현
"""
import pandas as pd
import numpy as np
import ast
from sklearn.cluster import KMeans

CSV_PATH     = r'C:\융융\2026\패션_비어플\최종2411022.csv'
OUTPUT_PATH  = r'C:\융융\2026\패션_비어플\bottom_rgb_list.csv'

df = pd.read_csv(CSV_PATH)
df['TOP_RGB']    = df['TOP_RGB'].apply(ast.literal_eval)
df['BOTTOM_RGB'] = df['BOTTOM_RGB'].apply(ast.literal_eval)

top_arr    = np.array(df['TOP_RGB'].tolist())
bottom_arr = np.array(df['BOTTOM_RGB'].tolist())

K = 200
print(f'KMeans k={K} 클러스터링 중...')
km = KMeans(n_clusters=K, n_init=10, random_state=42)
labels = km.fit_predict(top_arr)

rows = []
for c in range(K):
    mask = labels == c
    if not mask.any():
        continue
    rep_color  = tuple(map(int, km.cluster_centers_[c]))
    bottom_list = [tuple(b) for b in bottom_arr[mask]]
    rows.append({'Top_RGB_Cluster': c,
                 'Representative_Color': rep_color,
                 'Bottom_RGB_List': bottom_list})

out = pd.DataFrame(rows)
out.to_csv(OUTPUT_PATH, index=False)
print(f'완료: {len(out)}개 클러스터 → {OUTPUT_PATH}')
