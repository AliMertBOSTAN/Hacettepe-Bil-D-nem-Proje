import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt

def grayscale(_img):
    gray_image = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)[1]

    return thresh

def findPoints(thresh):
    sift = cv2.SIFT_create(contrastThreshold=0.08, edgeThreshold=4)
    kp = sift.detect(thresh, None)
    # img = cv2.drawKeypoints(thresh, kp, thresh, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return kp

def findAngle(kp1, kp2, kp3):
    x1, y1 = kp1.pt
    x2, y2 = kp2.pt
    x3, y3 = kp3.pt
    # print(x1,y1)
    # print(x2,y2)
    # print(x3,y3)

    vec1 = (x1 - x2, y1 - y2)
    vec2 = (x3 - x2, y3 - y2)

    dot_prod = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    # print(f"Vector 1 Magnitude: {mag1}, Vector 2 Magnitude: {mag2}")

    if mag1 == 0 or mag2 == 0:
        return 0  
    
    cos_angle = max(-1, min(1, dot_prod / (mag1 * mag2)))
    angle = math.degrees(math.acos(cos_angle))
    return angle

def apply_threshold(angles, threshold=3):
    filtered_angles = [angles[0]]

    for i in range(1, len(angles)):
        if abs(angles[i] - filtered_angles[-1]) > threshold:
            filtered_angles.append(filtered_angles[-1])
        else:
            filtered_angles.append(angles[i])

    return filtered_angles

def draw_keypoints_with_labels(img, keypoints):
    labeled_img = img.copy()
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        label = f"kp{i}"
        cv2.circle(labeled_img, (x, y), 5, (0, 255, 0), -1)  # Keypoint'leri çiz
        cv2.putText(labeled_img, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Etiket ekle
    return labeled_img

def main():
    cap = cv2.VideoCapture('./basket.mp4')
    pTime = 0
    angles = []
    frame_count = 0

    while cap.isOpened():
        cTime = time.time()
        success, img = cap.read()
        if not success:
            print("Video okuma hatası...")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps = 1 / (cTime - pTime)
        pTime = cTime

        h, w, c = img.shape

        cv2.rectangle(img, (0,0), (100,150), (0, 0, 0), -1)
        cv2.rectangle(img, (w,h) , (0,h-433), (0, 0, 0), -1)
        cv2.rectangle(img, (0,0) , (w,45), (0, 0, 0), -1)
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        thres = grayscale(img)
        keypoints = findPoints(thres)

        labeled_img = draw_keypoints_with_labels(thres, keypoints)
        cv2.imshow("Keypoints", labeled_img)

        unique_kps = []
        for kp in keypoints:
            if all(math.hypot(kp.pt[0] - ukp.pt[0], kp.pt[1] - ukp.pt[1]) > 1e-5 for ukp in unique_kps):
                unique_kps.append(kp)

        if len(unique_kps) >= 4:
            angle = findAngle(unique_kps[1], unique_kps[2], unique_kps[3])
            if angle is not None:
                angles.append(angle)  # Açıları kaydet
                print(f"Frame {frame_count}: Açı: {angle:.2f} derece")
            
            frame_count += 1
            
        # cv2.imshow("Point Detction", thres)

    cap.release()
    cv2.destroyAllWindows()

    thresholded_angles = apply_threshold(angles, threshold=3)


    plt.figure(figsize=(10, 6))
    plt.plot(angles, label="Açı Değerleri", linewidth=2)
    plt.plot(thresholded_angles, label="Thresholded Açı Değerleri", linewidth=2, linestyle='--')
    plt.title("Video Boyunca Açı Değişimi", fontsize=16)
    plt.xlabel("Frame Sayısı", fontsize=14)
    plt.ylabel("Açı (Derece)", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

    