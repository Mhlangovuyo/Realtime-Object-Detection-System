import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)


# Start video capture (0 = default webcam)
cap = cv2.VideoCapture(0)

print("Starting webcam... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)

    # Render the results on the frame
    annotated_frame = results.render()[0]

    # Display the frame
    cv2.imshow("YOLOv5 Webcam Detection", annotated_frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
