# Import necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# CAMERA
# Initialize camera
camera = PiCamera()
# Set resolution
camera.resolution = (640, 480)
# Set frames/sec
camera.framerate = 32
# Generate a 3D array and store the capture in it
raw_capture = PiRGBArray(camera, size = (640,480))

# OBJECT DETECTION
# Object detection threshold
thres = 0.5
# Coco-dataset names import them automatically into an array
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# Configuration file
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
# Create the model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Wait to let the camera warmup
time.sleep(0.1)

for frame in camera.capture_continuous(raw_capture, format="bgr"):

    img = frame.array
    print(img)
    #Send the picture to the model perform detection
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds, bbox)

    #Check that ot has detected something and that it ain't empty
    if len(classIds) != 0:
        #Create bounding box and write name
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=3)
            if classId<=82:
                #Print detection name
                cv2.putText(img,classNames[classId-1],(box[0]+10,box[1]+30),
                           cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                #Print detection value
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+150,box[1]+30),
                           cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
    cv2.imshow('Output', img)
    
    raw_capture.truncate(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cv2.destroyAllWindows()