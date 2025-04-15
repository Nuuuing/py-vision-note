import cv2
import numpy as np

# 1. 이미지 로드
img = cv2.imread("./assets/scramble.jpg")
if img is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")

h, w = img.shape[:2]

# 2. 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 하단 절반 추출
bottom_half = gray[h//2:, :]  # 아래 절반

# 4. 밝은 부분만 추출 - 100이상
mask = np.zeros_like(gray, dtype=np.uint8)
mask[h//2:, :] = np.where(bottom_half > 100, 255, 0)

# 5. 히트맵 생성
heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

# 6. 원본 이미지와 히트맵 합성
# → 강조할 부분만 합성, 나머지는 원본 유지
blended = img.copy()
highlight_area = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
blended = np.where(mask[..., None] > 0, highlight_area, img)  # 밝은 부분만 교체

# 7. 저장
cv2.imwrite("./assets/heatmap/highlight_mask.png", mask)
cv2.imwrite("./assets/heatmap/highlight_heatmap.png", heatmap)
cv2.imwrite("./assets/heatmap/blended_highlighted.png", blended)

print("생성 완료")