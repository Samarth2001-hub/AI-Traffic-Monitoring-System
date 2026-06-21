from ultralytics import YOLO
import cv2
import csv
from datetime import datetime
import easyocr

# Load YOLO model
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])

# Video path
video_path = "data/traffic.mp4"

cap = cv2.VideoCapture(video_path)
csv_file = open("traffic_data.csv", "w", newline="")

writer = csv.writer(csv_file)

writer.writerow([
    "Time",
    "Cars",
    "Motorcycles",
    "Buses",
    "Trucks",
    "Total",
    "Density"
])

# Vehicle class IDs
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    results = model(frame, conf=0.25)

    # Counters
    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls]

            # Count vehicles
            if cls == CAR:
                car_count += 1

            elif cls == MOTORCYCLE:
                motorcycle_count += 1

            elif cls == BUS:
                bus_count += 1

            elif cls == TRUCK:
                truck_count += 1

            # Draw only vehicle classes
            if cls in [CAR, MOTORCYCLE, BUS, TRUCK]:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                

    total = (
        car_count +
        motorcycle_count +
        bus_count +
        truck_count
    )
    # Traffic Density Analysis

    if total < 10:
        density = "LOW"
        alert = "NORMAL TRAFFIC"

    elif total < 20:
        density = "MEDIUM"
        alert = "MODERATE TRAFFIC"

    else:
        density = "HIGH"
        alert = "HEAVY TRAFFIC ALERT"
    current_time = datetime.now().strftime("%H:%M:%S")

    writer.writerow([
        current_time,
        car_count,
        motorcycle_count,
        bus_count,
        truck_count,
        total,
        density
    ])

    # Display counts
    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {motorcycle_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {bus_count}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {truck_count}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Total Vehicles: {total}",
        (10, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )
    cv2.putText(
    frame,
    f"Traffic Status: {density}",
    (10, 200),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 0, 0),
    2
)
    cv2.putText(
        frame,
        f"ALERT: {alert}",
        (10, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "Traffic Monitoring System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
csv_file.close()

cap.release()
cv2.destroyAllWindows()