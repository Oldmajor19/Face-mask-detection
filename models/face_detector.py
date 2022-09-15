# from ast import arg
import numpy as np
import argparse
import imutils
import time
import cv2
from imutils.video import VideoStream

# define the arguements
ap = argparse.ArgumentParser()

ap.add_argument('-p', '--prototxt', required=True, help='path to protox file')

ap.add_argument('-c','--confidence', type=float,default=0.5, help='min probability to filter weak detections')

ap.add_argument('-m', '--model', required=True, help='path to model')

args = vars(ap.parse_args())

print('[information] loading model.....')

net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

print('[information] starting video stream.....')

# vs = VideoStream(scr=0).start()
# time.sleep(1)

vs = cv2.VideoCapture(0)
time.sleep(2)

detected_objects = []
# loop over the frames from the video stream
while True:
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=800)
    image = frame.copy()
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                    mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
    
	# pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:2f}%".format(confidence*100)
            y = startY - 10 if startX - 10 > 10 else startY+10
            cv2.rectangle(frame, (startX,startY), (endX,endY),(0,0,255),2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


# do a bit of cleanup
cv2.destroyAllWindows()