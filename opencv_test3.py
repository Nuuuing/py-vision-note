import cv2

# 원본 이미지 로딩 (컬러)
img = cv2.imread("./assets/example.jpg", cv2.IMREAD_COLOR)
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 그레이스케일 + 블러 + Canny 엣지 검출
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

combined = cv2.addWeighted(img, 0.8, edges_colored, 0.2, 0)
# 첫번째 이미지 비율, 두번째 이미지 비율, 전체 결과 더할 값
cv2.imwrite("./assets/output_combined.jpg", combined)
print("✔ 윤곽선과 원본을 겹친 이미지 저장 완료!")