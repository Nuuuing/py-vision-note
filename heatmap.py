import cv2
import numpy as np

img = cv2.imread("./assets/scramble.jpg")

# 2. 이미지 로드 확인
if img is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다. 경로를 확인하세요.")

# 3. 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("./assets/heatmap/converted_gray.png", gray)
print("그레이스케일변환")

# 4. 히트맵 생성
# applyColorMap은 입력이 반드시 uint8 (0~255)이여야 함
gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
color_heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
cv2.imwrite("./assets/heatmap/color_heatmap.png", color_heatmap)
print("히트맵 생성")

# 5. 히트맵과 원본 크기 맞추기 (혹시 크기 다를 경우)
if color_heatmap.shape != img.shape:
    color_heatmap = cv2.resize(color_heatmap, (img.shape[1], img.shape[0]))

# 6. 히트맵과 원본을 합성
blended = cv2.addWeighted(img, 0.5, color_heatmap, 0.5, 0)
cv2.imwrite("./assets/heatmap/blended_heatmap.png", blended)
print("히트맵+원본 합성")