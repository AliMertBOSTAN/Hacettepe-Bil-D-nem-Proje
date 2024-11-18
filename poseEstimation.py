import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import toml
from pathlib import Path

# Sports2D klasör yolunu ekle
sys.path.append(os.path.abspath('/Users/alimertbostan/Documents/GitHub/hardhat burte force/Hacettepe-Bil-D-nem-Proje'))
from Sports2D import process

# MediaPipe Pose Estimator
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Webcam capture
cap = cv2.VideoCapture(1) # macbook ta ana webcam 1 olduğu için "1" verdim

# Sports2D için gerekli yapılandırma bilgisi
config_dict = toml.load("Config_demo.toml")

# result_dir değerini Path olarak ayarlayın
result_dir = Path(config_dict['project']['result_dir'])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for natural viewing
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe Pose Processing
    results = pose.process(rgb_frame)
    mediapipe_frame = frame.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            mediapipe_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Sports2D Pose Processing using process_fun
    video_file = "webcam"  # Canlı video olduğunu belirtmek için "webcam" kullanıyoruz
    time_range = (0, 1)  # Anlık olarak işlem yapacağız
    frame_rate = 30  # Varsayılan FPS değeri

    sports2d_landmarks = process.process_fun(config_dict, video_file, time_range, frame_rate, result_dir)
    for lm in sports2d_landmarks:
        cv2.circle(frame, (int(lm[0]), int(lm[1])), 5, (0, 255, 0), -1)
    
    # Display both frames
    cv2.imshow('MediaPipe Pose', mediapipe_frame)
    cv2.imshow('Sports2D Pose', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
