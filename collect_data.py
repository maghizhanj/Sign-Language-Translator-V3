import cv2
import csv
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

label = input("Enter sign label (e.g., A, B, C, hello): ")

print(f"Recording sign '{label}'...")
print("Press 'q' to stop recording")

with open("sign_data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        success, img = cap.read()

        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            landmarks = hand["lmList"]
            row = []
            x0 = landmarks[0][0]
            y0 = landmarks[0][1]

            for lm in landmarks:
                row.append(lm[0] - x0)
                row.append(lm[1] - y0)

            row.append(label)
            writer.writerow(row)

            cv2.putText(img, f"Recording {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Collect Data", img)

        key = cv2.waitKey(50)
        if key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Data saved to sign_data.csv")
