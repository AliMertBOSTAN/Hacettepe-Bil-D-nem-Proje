import mediapipe as mp
import numpy as np
import cv2
import time
import sys

# Import Sports2D; adjust the import as needed based on the library structure
sys.path.append('./Sports2D/Sports2D')
from Sports2D import Sports2D  # Adjust this import if needed

# Define the Sports2D detector class
class sports2dDetector():
    def __init__(self):
        # Initialize Sports2D (replace with actual initialization as needed)
        self.sports2d = Sports2D()
    
    def findPose(self, img):
        # Assuming Sports2D has an estimate_pose function that works with frames
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_result = self.sports2d.estimate_pose(frame_rgb)  # Adjust based on actual method
        # Overlay or process the results as necessary
        # This is a placeholder - add drawing code here if Sports2D provides coordinates to overlay
        
        return img  # Return the frame with the pose drawn if applicable

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
    cap = cv2.VideoCapture(1)
    pTime = 0
    detector = poseDetector()
    sports2d_detector = sports2dDetector()  # Initialize Sports2D detector

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Kayıt Hatası....")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        cTime = time.time()

        # Run MediaPipe pose detection
        img = detector.findPose(img, True)
        lmList = detector.findPosition(img, True)

        # Run Sports2D pose detection
        img = sports2d_detector.findPose(img)

        # Display FPS
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the combined result
        cv2.imshow("Pose Detection (MediaPipe + Sports2D)", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
