import cv2

def capture_photo():
    a = 0
    # 카메라 열기 (0은 기본 카메라)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("스페이스바를 눌러 사진을 찍고, ESC 키를 눌러 종료하세요.")

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 읽을 수 없습니다.")
            break

        # 화면에 보여주기
        cv2.imshow("Camera", frame)

        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키
            break
        elif key == 32:  # 스페이스바
            # 사진 저장
            cv2.imwrite(f"./assets/logi_example{a}.jpg", frame)
            a += 1
            print(f"사진이 저장되었습니다")

    # 카메라 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

# 사진 촬영 함수 호출
capture_photo()
