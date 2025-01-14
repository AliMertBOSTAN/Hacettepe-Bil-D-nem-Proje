import numpy as np
import cv2
import time
import math
import matplotlib.pyplot as plt

# ORB kullanılıyor
orb = cv2.ORB_create()

def grayscale(_img):
    # Gri tonlama ve threshold
    gray_image = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)
    return thresh

def findAngle(p1, p2, p3):
    # Üç nokta arasındaki açıyı hesapla
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    vec1 = (x1 - x2, y1 - y2)
    vec2 = (x3 - x2, y3 - y2)

    dot_prod = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = math.hypot(*vec1)
    mag2 = math.hypot(*vec2)

    if mag1 == 0 or mag2 == 0:
        return 0

    cos_angle = max(-1, min(1, dot_prod / (mag1 * mag2)))
    angle = math.degrees(math.acos(cos_angle))
    return angle

def findKeypointsAndDescriptors(thresh):
    kp, des = orb.detectAndCompute(thresh, None)
    return kp, des

def draw_points_and_angle(img, points, angle):
    # Noktaları çiz
    for i, point in enumerate(points):
        cv2.circle(img, point, 5, (0, 255, 0), -1)
        cv2.putText(img, f"P{i+1}", (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Üç nokta arasına çizgi çiz
    cv2.line(img, points[0], points[1], (255, 0, 0), 2)
    cv2.line(img, points[1], points[2], (255, 0, 0), 2)

    # Açıyı görüntüye yaz
    cv2.putText(img, f"Aci: {angle:.2f} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def match_previous_points(prev_points, curr_keypoints):
    """
    Önceki karedeki noktaları, mevcut karedeki keypoint'lerle eşleştirir.
    """
    matched_points = []
    for prev_pt in prev_points:
        min_distance = float('inf')
        closest_pt = None

        for kp in curr_keypoints:
            curr_pt = (int(kp.pt[0]), int(kp.pt[1]))
            distance = math.hypot(prev_pt[0] - curr_pt[0], prev_pt[1] - curr_pt[1])
            
            if distance < min_distance:
                min_distance = distance
                closest_pt = curr_pt

        matched_points.append(closest_pt)
    
    return matched_points

def main():
    cap = cv2.VideoCapture('./basket.mp4')
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pTime = 0
    angles = []
    frame_count = 0

    prev_points = None  # Önceki karede seçilen noktalar

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Video okuma hatası...")
            break

        # FPS hesaplama
        cTime = time.time()
        fps_display = int(1 / (cTime - pTime + 1e-5))
        pTime = cTime

        h, w, c = img.shape

        cv2.rectangle(img, (0,0), (100,150), (0, 0, 0), -1)
        cv2.rectangle(img, (w,h) , (0,h-433), (0, 0, 0), -1)
        cv2.rectangle(img, (0,0) , (w,45), (0, 0, 0), -1)
        thres = grayscale(img)
        keypoints, descriptors = findKeypointsAndDescriptors(thres)


        # Nokta seçim işlemi
        if len(keypoints) >= 3:
            if prev_points is None:
                # İlk karede 3 keypoint seçilir
                prev_points = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints[:3]]
            else:
                # Sonraki karelerde önceki noktalarla en yakın eşleşme yapılır
                prev_points = match_previous_points(prev_points, keypoints)

            # Açı hesaplama
            if len(prev_points) == 3:
                angle = findAngle(prev_points[0], prev_points[1], prev_points[2])
                angles.append(angle)
                print(f"Frame {frame_count}: Açı: {angle:.2f} derece")

                # Noktaları ve açıyı çiz
                draw_points_and_angle(img, prev_points, angle)

        frame_count += 1

        # FPS göster
        cv2.putText(img, f'FPS: {fps_display}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Keypoints and Angle", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Açı değişimini grafikle göster
    plt.figure(figsize=(10, 6))
    plt.plot(angles, label="Açı Değerleri", linewidth=2)
    plt.title("Video Boyunca Açı Değişimi", fontsize=16)
    plt.xlabel("Frame Sayısı", fontsize=14)
    plt.ylabel("Açı (Derece)", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
