import cv2

# 1. 이미지 불러오기
input_path = "./assets/example.jpg"
# img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # 흑백으로 읽기
img = cv2.imread(input_path, cv2.IMREAD_COLOR)

if img is None:
    print("이미지 불러오기 실패!")
    exit()

# 2. 윤곽선 추출 (Canny)
edges = cv2.Canny(img, threshold1=50, threshold2=200)

# 3. 히트맵으로 변환 (COLORMAP_JET or others)
heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

# 4. 결과 저장
cv2.imwrite("./assets/output_edges2.jpg", edges)
cv2.imwrite("./assets/output_heatmap2.jpg", heatmap)

print("✔ 윤곽선 및 히트맵 이미지 저장 완료!")