import cv2

import numpy as np

from rotate1 import getMovement

# from picamera2 import Picamera2

import tensorflow as tf

from detect1 import run

# import RPi.GPIO as GPIO

# import servomotor as sm

from sendingMail import sendingMail
from tensorflow.keras.preprocessing import image

import time

# from pump import pump1

# model = tf.keras.models.load_model('data_augmentation.h5')

# GPIO.setmode(GPIO.BCM)

# pump_pin = 24

# GPIO.setup(pump_pin, GPIO.OUT,initial=GPIO.LOW)

# picam2=Picamera2()

# picam2.configure(picam2.create_preview_configuration(

#   main={"size": (2304,1296)}, # High resolution for processing

#   lores={"size": (1080, 720)} # Lower resolution for preview

# ))

cal=0

prev_frame = None

# picam2.start()

# C:\Users\Pranay\OneDrive\Desktop\test3\yolov5-fire-detection\input.mp4

video_path = r"C:\Users\Pranay\OneDrive\Desktop\test3\yolov5-fire-detection\input.mp4"
cap = cv2.VideoCapture(video_path)

# cap = cv2.VideoCapture("yolov5-fire-detection/input.mp4")

while(True):

#   frame=picam2.capture_array()

  ret, frame = cap.read()

  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

  if prev_frame is None:



    prev_frame = gray_frame



    continue

  frame_diff = cv2.absdiff(prev_frame, gray_frame)



  _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

  prev_frame = gray_frame



  movement_detected = np.sum(thresh) > 10000 # Adjust threshold as needed

  frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

  # frame_bgr=frame

  if movement_detected:

     



    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)



    l_b = np.array([10, 139, 234])

    u_b = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, l_b, u_b)



    res = cv2.bitwise_and(frame, frame, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:

      largest_contour = max(contours, key=cv2.contourArea)

      x, y, w, h = cv2.boundingRect(largest_contour)

      cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

      print(f"Largest bounding box coordinates: x={x}, y={y}, w={w}, h={h}")

      # cv2.imshow("Mask", mask)

      cv2.imshow("Result", res)

      cv2.imwrite('test.jpg',frame_bgr)

      fr=cv2.resize(frame_bgr,(64,64))

      tr= image.img_to_array(fr)

      print(x,y)

      # tr = np.expand_dims(tr, axis=0)

      # tr = tr / 255.0 # Normalize if the model expects it

      # Step 3: Make the prediction

      # predictions = model.predict(tr)

      # print(predictions)

      # if(predictions[0][1]>0.1):

      print("fire detected by opencv")

      if w>20 and h>20:

        ans=run(

        weights= "../model/yolov5s_best.pt", # your model path

        source= "test.jpg", # input video file

        data= "data/coco128.yaml", # dataset.yaml path

        imgsz=(640, 640), # inference size

        conf_thres=0.25, # confidence threshold

        iou_thres=0.45, # NMS IOU threshold

        max_det=1000, # maximum detections per image

        device="", # use default device (auto-select)

        view_img=False, # don't show images during inference

        save_txt=False, # don't save labels

        save_csv=False, # don't save CSV

        save_conf=False, # don't save confidences

        save_crop=False, # don't save cropped boxes

        nosave=False, # don't skip saving images

        classes=None, # detect all classes

        agnostic_nms=False, # class-agnostic NMS

        augment=False, # no augmentation

        visualize=False, # no feature visualization

        update=False, # don't update model

        project= "runs/detect", # directory to save results

        name="exp", # experiment name

        exist_ok=False, # don't overwrite existing experiment

        line_thickness=3, # bounding box line thickness

        hide_labels=False, # show labels

        hide_conf=False, # show confidence scores

        half=False, # use FP16 precision

        dnn=False, # use OpenCV DNN for ONNX inference

        vid_stride=1, # process every frame in the video

        )

        if not ans:

          cal=0

          print("fire is not detected")

        #   sm.kit.servo[0].angle = 90

        #   sm.kit.servo[1].angle = 20

          # GPIO.output(pump_pin, GPIO.LOW)

        else:

          print("fire confirmed start spraying water")

          cv2.imwrite('image.jpg',frame_bgr)
          
          sendingMail()

          getMovement()

          # moving function

          cal+=1

          if cal==1:

            xmin=ans[0][2]

            xmax=ans[0][4]

            ymin=ans[0][3]

            ymax=ans[0][5]

            cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)

            # sm.servo2(int((ymax+ymin)/2))

            # sm.servo1(int((xmin+xmax)/2))

            # pump1()

            # GPIO.output(pump_pin, GPIO.HIGH)



    else:

      # cv2.imshow("Mask", np.zeros_like(frame))

      cv2.imshow("Result", np.zeros_like(frame))

    #   sm.kit.servo[0].angle = 90

    #   sm.kit.servo[1].angle = 20

      time.sleep(2)

      cal=0

      # GPIO.output(pump_pin, GPIO.LOW)

      p=0

  cv2.imshow("Original", frame_bgr)

  key = cv2.waitKey(1)

  if key == 27:

    break

# picam2.stop()

# GPIO.output(pump_pin, GPIO.LOW)

# sm.kit.servo[0].angle = 90

# sm.kit.servo[1].angle = 20

cv2.destroyAllWindows()

# GPIO.cleanup()