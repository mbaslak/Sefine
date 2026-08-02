import cv2 as cv
import numpy as np
from ultralytics import YOLO
#from pathlib import Path





def write_detected_objects(boxes, frame, model):

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        confidence = float(box.conf[0])

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y1) / 2)

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 50), 1)

        label = f"{class_name} %{confidence*100:.0f}"

        cv.putText(frame, label, (x1, max(y1 - 10, 20)), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 100), 2)

        cv.circle(frame, (center_x, center_y), 3, (0, 0, 200), -1)

        print(f"Detected: {class_name} | Confidence: %{confidence * 100:.1f} | Center: ({center_x}, {center_y})")




def main():

    

    stream = cv.VideoCapture(0)

    if not stream.isOpened():
        print("The camera could not be opened.")
        exit()

    output_1 = cv.VideoWriter("test_1.mp4", fourcc=cv.VideoWriter_fourcc('m', 'p', '4', 'v'), fps=stream.get(cv.CAP_PROP_FPS), frameSize=(int(stream.get(3)), int(stream.get(4))))
    output_2 = cv.VideoWriter("test_2.mp4", fourcc=cv.VideoWriter_fourcc('m', 'p', '4', 'v'), fps=stream.get(cv.CAP_PROP_FPS), frameSize=(int(stream.get(3)), int(stream.get(4))))

    

    model = YOLO("best_m.engine", task="detect")

    try:

        while True:

            ret, frame = stream.read()

            if not ret:
                print("An error occured during the streaming.")
                exit()

            #frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)



            height = frame.shape[0]
            width = frame.shape[1]

            xf = int(width / 2)
            yf = int(height / 2)

            M = cv.getRotationMatrix2D((xf, yf), 90, 1.0)
            frame = cv.warpAffine(frame, M, (width, height))

            frame = cv.putText(frame, "+", (xf, yf), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=4)

            results = model(frame, verbose=False)

            annotated = results[0].plot()

            write_detected_objects(results[0].boxes, frame, model)
            output_1.write(frame)
            output_2.write(annotated)

            inp = cv.waitKey(1)
            if inp == ord('d'):
                #cv.destroyAllWindows()
                break

    except KeyboardInterrupt as e:
        print(f"An exception has been thrown: {e}")
        exit(1)

    finally:
        stream.release()
        output_1.release()
        output_2.release()
        cv.destroyAllWindows()
        print("All sources have been freed.")

if __name__ == "__main__":

    main()
    