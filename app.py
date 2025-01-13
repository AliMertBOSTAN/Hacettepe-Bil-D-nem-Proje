import mediapipe as mp
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.75, trackCon=0.75):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
                                     smooth_landmarks=self.smooth, 
                                     min_detection_confidence=self.detectionCon, 
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

def main():
    cap = cv2.VideoCapture("./basket.mp4")
    pTime = 0
    detector = poseDetector()
    fps_values = []  # FPS değerlerini saklamak için liste


    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Kayıt Hatası....")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        cTime = time.time()
        img = detector.findPose(img, True)
        lmList = detector.findPosition(img, True)

        fps = 1 / (cTime - pTime)
        pTime = cTime
        fps_values.append(fps)  # FPS değerini listeye ekle


        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Pose Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

     # FPS grafiği
    plt.figure(figsize=(12, 6))
    plt.plot(fps_values, label="FPS Değerleri", linewidth=2, color='orange')
    plt.title("Video Boyunca FPS Değişimi", fontsize=16)
    plt.xlabel("Frame Sayısı", fontsize=14)
    plt.ylabel("FPS", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
