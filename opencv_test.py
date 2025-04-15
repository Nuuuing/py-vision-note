import cv2

# 이미지 파일 경로
input_path = "./assets/example.jpg"
blur_path = "./assets/blur_saved.jpg"

# 이미지 읽기 (컬러로)
img = cv2.imread(input_path, cv2.IMREAD_COLOR)

# 읽기에 실패한 경우 처리
if img is None:
    print("이미지 로딩 실패! 경로를 확인하세요.")
else:
    print("이미지 로딩 성공! 크기:", img.shape)

    # 이미지 저장
    # cv2.imwrite(output_path, img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    cv2.imwrite(blur_path, blur)
    print(f"이미지 저장 완료 → {output_path}")

