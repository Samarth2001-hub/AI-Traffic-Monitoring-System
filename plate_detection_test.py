from ultralytics import YOLO
import cv2

plate_model = YOLO("data/models/license_plate.pt")
image = cv2.imread("data/images.png")

results = plate_model(image)

for r in results:

    for box in r.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

cv2.imshow("Plate Detection", image)

cv2.waitKey(0)

cv2.destroyAllWindows()