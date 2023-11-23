import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 빈 dataframe 생성
df = pd.DataFrame()

# 스쿼트 카운트 및 상태를 추적하는 변수
squat_count = 0
is_down = False

# 마지막 카운트가 증가한 시간을 추적하는 변수
last_count_time = 0

#벡터 사이의 각도를 계산하는 함수
def calculate_angle(a, b):
    inner_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = inner_product / (norm_a * norm_b)
    angle = np.arccos(cos_theta)
    return math.degrees(angle)

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            # 동영상을 불러올 경우는 'continue' 대신 'break'를 사용합니다.
            continue

        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        #pose data 저장
        #빈 리스트 x 생성
        x=[]

        #k = landmarks 개수 및 번호
        #result.pose_landmarks[k].x/y/z/visiblity로 k번째 landmarks의 정보를 가져올 수 있다.
        if results.pose_landmarks:
            # for k in range(33):
            #     landmark = results.pose_landmarks.landmark[k]
            #     x.append(landmark.x)
            #     x.append(landmark.y)
            #     x.append(landmark.z)
            #     x.append(landmark.visibility)

            #     print(f"Landmark {k}: x = {landmark.x}, y = {landmark.y}, z = {landmark.z}, visibility = {landmark.visibility}")
            # necessary_node = [11, 12, 23, 24]
            # for node in necessary_node:
            #     landmark = results.pose_landmarks.landmark[11]
            #     x.append(landmark.x)
            #     x.append(landmark.y)
            #     x.append(landmark.z)
            #     x.append(landmark.visibility)
            #     print(f"x = {landmark.x}, y = {landmark.y}, z = {landmark.z},")

            # 필요한 랜드마크의 좌표를 추출
            # 11-left shoulder, 12-right shoulder, 23-left hip, 24-right hip
            landmarks = results.pose_landmarks.landmark
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            left_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
            right_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])

            # 대퇴골을 나타내는 랜드마크
            hip = np.array([landmarks[24].x, landmarks[24].y])
            knee = np.array([landmarks[26].x, landmarks[26].y])

            # 대퇴골 벡터
            thigh_vector = knee - hip

            # 벡터 계산
            shoulder_vector = right_shoulder - left_shoulder
            hip_vector = right_hip - left_hip

            # 대퇴골 각도 계산 (y-좌표만 사용)
            angle = np.arctan2(thigh_vector[1], thigh_vector[0]) * 180 / np.pi

            # 척추 각도 계산
            spine_angle = calculate_angle(shoulder_vector, hip_vector)
            print(f"Angle: {spine_angle}")  

            # 스쿼트 상태 감지 및 카운트 증가
            if angle > 45:  # 하강 구간임계값 설정
                is_down = True
            elif angle < 0 and is_down:  # 상승 구간임계값 설정
                is_down = False
                current_time = time.time()
                if current_time - last_count_time > 1:  # 1초 간격 확인
                    squat_count += 1
                    last_count_time = current_time          

            #list x를 dataframe으로 변경함
        tmp = pd.DataFrame(x).T

        #dataframe에 정보 쌓아주기
        #33개 landmarks의 132개 정보가 dataframe에 담김
        df = pd.concat([df, tmp])

        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        image = cv2.flip(image, 1)
        
        
         # 스쿼트 카운트와 상태를 표시
        cv2.putText(image, f'Squat Count: {squat_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'Status: {"UP" if is_down else "DOWN"}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        image = cv2.flip(image, 1)
        
        # 보기 편하게 이미지를 좌우 반전합니다.
        cv2.imshow('EgoGym', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
        
    
cap.release()
cv2.destroyAllWindows()