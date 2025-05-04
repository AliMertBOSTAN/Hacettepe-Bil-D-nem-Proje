import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt
import mediapipe as mp
import os  # Dosya kontrolü için eklendi

# OpenCV SIFT İşlemleri
sift = cv2.SIFT_create(contrastThreshold=0.08, edgeThreshold=4)

def grayscale(_img):
    gray_image = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)[1]
    return thresh

def findKeypointsAndDescriptors(thresh):
    kp, des = sift.detectAndCompute(thresh, None)
    return kp, des

# MediaPipe İşlemleri
class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.75, trackCon=0.75):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=mode, 
                                     smooth_landmarks=smooth, 
                                     min_detection_confidence=detectionCon, 
                                     min_tracking_confidence=trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

# OpenCV Ana Fonksiyonu
def run_opencv():
    if not os.path.exists('./basket.mp4'):
        print("HATA: basket.mp4 dosyası bulunamadı!")
        return
    cap = cv2.VideoCapture('./basket.mp4')
    while cap.isOpened():
        success, img = cap.read()
        if not success or img is None:
            break
        thres = grayscale(img)
        keypoints, descriptors = findKeypointsAndDescriptors(thres)
        img_with_kp = cv2.drawKeypoints(thres, keypoints, None)
        if img_with_kp is not None:
            cv2.imshow("OpenCV Keypoints", img_with_kp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# MediaPipe Ana Fonksiyonu
def run_mediapipe():
    if not os.path.exists('./basket.mp4'):
        print("HATA: basket.mp4 dosyası bulunamadı!")
        return
    cap = cv2.VideoCapture("basket.mp4")
    detector = poseDetector()
    while cap.isOpened():
        success, img = cap.read()
        if not success or img is None:
            break
        img = detector.findPose(img)
        if img is not None:
            cv2.imshow("MediaPipe Pose Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Ana Menü Fonksiyonu
def main():
    from threading import Thread
    opencv_thread = Thread(target=run_opencv)
    mediapipe_thread = Thread(target=run_mediapipe)

    opencv_thread.start()
    mediapipe_thread.start()

    opencv_thread.join()
    mediapipe_thread.join()

if __name__ == "__main__":
    main()
