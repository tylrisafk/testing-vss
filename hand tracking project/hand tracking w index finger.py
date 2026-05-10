import cv2
import mediapipe as mp

# 1. Setup the "Brain"
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=3, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 2. Setup the "Eyes"
cap = cv2.VideoCapture(0)

print("Hand Tracker is running... Press 'q' to quit.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # 3. Draw the landmarks if hands are found
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # This draws the dots and lines
            mp_draw.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger_tip = landmarks.landmark[8]
            h, w, c = img.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

    # 4. Show the window
    cv2.imshow("Engineering Project - Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()