import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture("Volleyball.mp4")

if not cap.isOpened():
    print("Error opening video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

width = 960
height = int(original_height * width / original_width)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    "Volleyball_Output.mp4",
    fourcc,
    fps,
    (width, height)
)

previous_position = None
trajectory = deque(maxlen=25)

print("Volleyball tracking started!")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (width, height))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=25,
        param1=100,
        param2=22,
        minRadius=3,
        maxRadius=16
    )

    chosen = None

    if circles is not None:

        circles = np.round(circles[0]).astype(int)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        candidates = []

        for x, y, r in circles:

            x1 = max(0, x-r)
            y1 = max(0, y-r)
            x2 = min(width, x+r+1)
            y2 = min(height, y+r+1)

            patch = hsv[y1:y2, x1:x2]

            if patch.size == 0:
                continue

            white = cv2.inRange(
                patch,
                np.array([0, 0, 150]),
                np.array([180, 110, 255])
            )

            white_ratio = cv2.countNonZero(white) / (
                patch.shape[0] * patch.shape[1]
            )

            if white_ratio > 0.25:
                candidates.append((x, y, r, white_ratio))

        if candidates:

            if previous_position is None:
                chosen = max(
                    candidates,
                    key=lambda c: c[3]
                )
            else:
                nearby = [
                    c for c in candidates
                    if np.hypot(
                        c[0] - previous_position[0],
                        c[1] - previous_position[1]
                    ) < 120
                ]

                if nearby:
                    chosen = min(
                        nearby,
                        key=lambda c: np.hypot(
                            c[0] - previous_position[0],
                            c[1] - previous_position[1]
                        )
                    )

    if chosen is not None:

        x, y, r, _ = chosen

        previous_position = (x, y)
        trajectory.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            r + 5,
            (255, 0, 255),
            2
        )

        cv2.circle(
            frame,
            (x, y),
            3,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            "VOLLEYBALL",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )

    for i in range(1, len(trajectory)):

        cv2.line(
            frame,
            trajectory[i - 1],
            trajectory[i],
            (0, 255, 0),
            2
        )

    status = "BALL DETECTED" if chosen is not None else "SEARCHING..."

    cv2.putText(
        frame,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Volleyball Tracking", frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Tracking completed!")
print("Output saved as Volleyball_Output.mp4")