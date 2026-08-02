import cv2 as cv
import numpy as np
from ultralytics import YOLO
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
import asyncio
from dataclasses import dataclass
#from pathlib import Path


#ef write_detected_objects(frame,):

@dataclass
class Coordinate:
    x: int
    y: int

@dataclass
class Yolo_object:
    confidence: float
    center: Coordinate
    class_id: int
    class_name: str
    vertices: list = [None, None]

xf = 0
yf = 0

target_center_x = -1
target_center_y = -1

is_locked = False
is_found = False

drone : System = None

detected_object = Yolo_object(0.0, None, None, None)



async def check_lock():
    global is_locked

    while True:

        if  is_found == True and abs(target_center_x - xf) <= 10:
            is_locked = True
        else:
            is_locked = False

        asyncio.sleep(2)
    

async def check_rotation():

    while True:

        if is_found == True and is_locked == False:
            rotate_to_target()

        asyncio.sleep(3)
        
async def rotate(is_right):

    if is_right == True:
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 36.0)
        )
    else:
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, -36.0)
        )

    

    while True:

        if is_locked == True:
            break

        asyncio.sleep(0.5)

    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    )  


async def go_ahead():

    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(
            forward_m_s=0.70,
            right_m_s=0.0,
            down_m_s=0.0,
            yawspeed_deg_s=0.0
        )
    )

    while True:

        edge_1 = abs(detected_object.vertices[0].x - detected_object.vertices[1].x)
        edge_2 = abs(detected_object.vertices[0].y - detected_object.vertices[1].y)

        area = edge_1 * edge_2

        if is_found == False or is_locked == False or area > 20000:
            break

        asyncio.sleep(1)

    
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(
            forward_m_s=0.0,
            right_m_s=0.0,
            down_m_s=0.0,
            yawspeed_deg_s=0.0
        )
    )
        

async def take_objects(boxes, model):

    global is_found
    global is_locked
    
    flag = False

    for box in boxes:
    
            x1, y1, x2, y2 = map(int, box.xyxy[0])
    
            confidence = float(box.conf[0])
    
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
    
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name == "orange_buoy" or class_name == "red_buoy" and confidence > 0.75:

                is_found = True
                flag = True

                detected_object.class_name = class_name
                detected_object.class_id = class_id
                detected_object.center = Coordinate(center_x, center_y)
                detected_object.confidence = confidence
                detected_object.vertices[0] = Coordinate(x1, y1)
                detected_object.vertices[1] = Coordinate(x2, y2)

                break

    if flag == False:
        is_found = False
        is_locked = False

        set_state = VelocityBodyYawspeed(
                forward_m_s=0.0,
                right_m_s=0.0,
                down_m_s=0.0,
                yawspeed_deg_s=0.0
        )

        await drone.offboard.set_velocity_body(set_state)

async def rotate_to_target():

    set_state = VelocityBodyYawspeed(
        forward_m_s=0.0,
        right_m_s=0.0,
        down_m_s=0.0,
        yawspeed_deg_s=0.0
    )

    await drone.offboard.set_velocity_body(set_state)

    # try:
    #     await drone.offboard.start()
    #     print("Offboard starting...")
    # except OffboardError as error:
    #     print(f"Offboard could not be opened: {error._result.result}")
    #     return

    if xf > detected_object.center.x:

        await rotate(is_right=False)

    else:
        await rotate(is_right=True)
        


    # cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 50), 1)

    # label = f"{class_name} %{confidence*100:.0f}"

    # cv.putText(frame, label, (x1, max(y1 - 10, 20)), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 100), 2)

    # cv.circle(frame, (center_x, center_y), 3, (0, 0, 200), -1)

    # print(f"Detected: {class_name} | Confidence: %{confidence * 100:.1f} | Center: ({center_x}, {center_y})")



def camera_process(stream, model, output_1, output_2):

    try:

        while True:

            ret, frame = stream.read()

            if not ret:
                print("An error occured during the streaming.")
                return

            #frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)



            height = frame.shape[0]
            width = frame.shape[1]

            #xf = int(width / 2)
            # yf = int(height / 2)

            M = cv.getRotationMatrix2D((xf, yf), 90, 1.0)
            frame = cv.warpAffine(frame, M, (width, height))

            results = model(frame, verbose=False)

            annotated = results[0].plot()

            frame = cv.putText(frame, "+", (xf, yf), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=4)

            #write_detected_objects(results[0].boxes, frame, model)
            output_1.write(frame)
            output_2.write(annotated)

            inp = cv.waitKey(1)
            if inp == ord('d'):
                #cv.destroyAllWindows()
                break

    except KeyboardInterrupt as e:
        print(f"An exception has been thrown: {e}")
        return

    except:
        print(f"An unknown error occured.")
        return

    finally:
        stream.release()
        output_1.release()
        output_2.release()
        cv.destroyAllWindows()
        print("All sources have been freed.")



async def main(drone_arg : System):

    drone = drone_arg

    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

    try:
        await drone.offboard.start()

    except OffboardError as error:
        print(f"Offboard mode could not be opened.: {error._result}")

    stream = cv.VideoCapture(0)

    if not stream.isOpened():
        print("The camera could not be opened.")
        return

    width = int(stream.get(3))
    height = int(stream.get(4))

    xf = width / 2
    yf = height / 2

    output_1 = cv.VideoWriter("test_1.mp4", fourcc=cv.VideoWriter_fourcc('m', 'p', '4', 'v'), fps=stream.get(cv.CAP_PROP_FPS), frameSize=(int(stream.get(3)), int(stream.get(4))))
    output_2 = cv.VideoWriter("test_2.mp4", fourcc=cv.VideoWriter_fourcc('m', 'p', '4', 'v'), fps=stream.get(cv.CAP_PROP_FPS), frameSize=(int(stream.get(3)), int(stream.get(4))))

    

    model = YOLO("best_m.engine", task="detect")

    
    await drone.offboard.stop()

