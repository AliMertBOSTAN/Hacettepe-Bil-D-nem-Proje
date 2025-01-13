import mediapipe as mp
import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt

angle = [] 

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
    
    def findeAngle(self, img, p1, p2, p3, draw=True):
        try:
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            x3, y3 = self.lmList[p3][1:]

            angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2, x1-x2))
            if angle < 0:
                angle = angle + 360


            if draw:
                cv2.line(img, (x1, y1),(x2,y2),(255,255,255),2)
                cv2.line(img, (x3, y3),(x2,y2),(255,255,255),2)

                cv2.circle(img, (x1,y1), 2,(0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x1,y1), 4,(255, 0, 0), 1)

                cv2.circle(img, (x2,y2), 2,(0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2,y2), 4,(255, 0, 0), 1)

                cv2.circle(img, (x3,y3), 2,(0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3,y3), 4,(255, 0, 0), 1)

            return  angle
        except:
            KeyError("Açı okuma Hatası")
            return 0

def findAngle(img, detector):
    img = cv2.flip(img, 1)
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)


    # angleSag = detector.findeAngle(img, 12, 14, 16)
    angleSol = detector.findeAngle(img, 23, 11, 13)
    per = np.interp(angleSol, (200,270),(0,100))

    # angle1 = detector.findeAngle(img, 13, 11, 23)
    # angle2 = detector.findeAngle(img, 14, 12, 24)

    angle.append(angleSol)
    return img

def apply_threshold(angles, threshold=3):
    filtered_angles = [angles[0]]

    for i in range(1, len(angles)):
        if abs(angles[i] - filtered_angles[-1]) > threshold:
            filtered_angles.append(filtered_angles[-1])
        else:
            filtered_angles.append(angles[i])

    return filtered_angles

def main():
    cap = cv2.VideoCapture("basket.mp4")
    pTime = 0
    detector = poseDetector()

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Kayıt Hatası....")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        cTime = time.time()
        # img = detector.findPose(img, True)
        # lmList = detector.findPosition(img, True)
        img = findAngle(img, detector)

        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Pose Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    thresholded_angles = apply_threshold(angle, threshold=5)
    cv2.destroyAllWindows()
    plt.figure(figsize=(10, 6))
    plt.plot(angle, label="Açı Değerleri", linewidth=2)
    plt.plot(thresholded_angles, label="Thresholded Açı Değerleri", linewidth=2, linestyle='--')
    plt.title("Video Boyunca Açı Değişimi", fontsize=16)
    plt.xlabel("Frame Sayısı", fontsize=14)
    plt.ylabel("Açı (Derece)", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
