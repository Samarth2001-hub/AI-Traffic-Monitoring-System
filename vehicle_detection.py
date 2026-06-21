from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video_path = "data/traffic.mp4"
cap = cv2.VideoCapture(video_path)

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame = cv2.resize(frame, (640, 360))

    results = model(frame, conf=0.25)
    annotated_frame = results[0].plot()

    cv2.imshow("Traffic Monitoring System", annotated_frame)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = model.names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                cv2.putText(frame, label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

    cv2.imshow("Traffic Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 