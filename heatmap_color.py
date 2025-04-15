import cv2
import numpy as np

# 1. 이미지 로드
img = cv2.imread("./assets/scramble.jpg")
if img is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")

# 2. 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./assets/gray.png", gray)

# 3. 히스토그램 평탄화
equalized = cv2.equalizeHist(gray)
cv2.imwrite("./assets/heatmap/equalized_gray.png", equalized)

# 4. 히트맵 적용 전: normalize (optional but safe)
gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
equalized_norm = cv2.normalize(equalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 5. 히트맵 생성
heatmap_orig = cv2.applyColorMap(gray_norm, cv2.COLORMAP_JET)
heatmap_equalized = cv2.applyColorMap(equalized_norm, cv2.COLORMAP_JET)

cv2.imwrite("./assets/heatmap/heatmap_original.png", heatmap_orig)
cv2.imwrite("./assets/heatmap/heatmap_equalized.png", heatmap_equalized)

# 6. 비교를 위한 합성 이미지 생성
blended_orig = cv2.addWeighted(img, 0.5, heatmap_orig, 0.5, 0)
blended_equalized = cv2.addWeighted(img, 0.5, heatmap_equalized, 0.5, 0)

cv2.imwrite("./assets/heatmap/blended_heatmap_original.png", blended_orig)
cv2.imwrite("./assets/heatmap/blended_heatmap_equalized.png", blended_equalized)

print("히스토그램 평탄화 및 히트맵 이미지 저장 완료!")
