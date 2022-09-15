from ast import While
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import cv2

modelFile =r"C:\Users\pc\Desktop\delete\caffe_model_for_dace_detection\res10_300x300_ssd_iter_140000.caffemodel"
configFile =r"C:\Users\pc\Desktop\delete\caffe_model_for_dace_detection\deploy.prototxt.txt"

#load caffe model
net = cv2.dnn.readNetFromCaffe( configFile, modelFile)
# load our mask_detection model
model = tf.keras.models.load_model('face_mask_model.h5')

# Define mask_label
mask_label = {0:'MASK INCORRECT',1:'MASK', 2:'NO MASK'}
color_label = {0:(0,255,255),1:(0, 255,0), 2:(255,0,0)}

vs = cv2.VideoCapture(0)
time.sleep(2)

detected_objects = []

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
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            y = startY - 10 if startX - 10 > 10 else startY+10



            face = image[startY:endY, startX:endX]
            resized_face = cv2.resize(np.array(face),(60,60))
            #resized_face = resized_face[:, :, :3]
            reshaped_face = np.reshape(resized_face,[1,60,60,3])/255.0
            face_result = model.predict(reshaped_face)
            print(mask_label[face_result.argmax()])
            print(face_result.argmax())

            cv2.rectangle(frame, (startX,startY), (endX,endY),(0,0,255),2)
            cv2.putText(frame, mask_label[face_result.argmax()], (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
    

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


# do a bit of cleanup
cv2.destroyAllWindows()