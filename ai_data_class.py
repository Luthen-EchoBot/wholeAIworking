class Gesture_Data:
    def __init__(self,class_name,ID,probability):
        self.class_name=class_name
        self.id=ID
        self.probability=probability

class Detection:
    def __init__(self,x,y,w,h,id,probability,estimated_distance,class_name):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id=id
        self.probability=probability
        self.dist=estimated_distance
        self.class_name=class_name

class AI_Data:
    def __init__(self,detections,gesture_data):
        self.detections=detections
        self.gesture_data=gesture_data

SOCKET_PORT = 50007
MAX_MSG_LEN = 16384
