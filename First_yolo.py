#based on CPU and opencv
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
coco_file= 'coco.names'
coco_classes = []
net_config = 'cfg/yolov3.cfg'
net_weights = 'cfg/yolov3.weights'
blob_size = 320
confidence_threshold = 0.4
nms_threshold = 0.3

# r=read, t=text mode, b=binary mode
with open(coco_file,'rt') as f:
    coco_classes = f.read().rstrip('\n').split('\n')

# generate different colors for different classes 
COLORS = np.random.random_integers(0, 255, size=(len(coco_classes), 3))

# put an object to cerate a darknet network
net = cv.dnn.readNetFromDarknet(net_config,net_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def findObjects(out_put, img):
    img_h,img_w,_ = img.shape
    
    #initialization
    bboxes = []
    class_ids =[]
    confidences = []

    for item in out_put:
        for detect_vector in item:
            scores = detect_vector[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                w,h =  int(detect_vector[2] * img_w ),int(detect_vector[3] * img_h)
                x,y = int((detect_vector[0]*img_w) - w/2), int((detect_vector[1]*img_h)-h/2)
                bboxes.append([x,y,w,h])
                class_ids.append(class_id)
                confidences.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bboxes,confidences,confidence_threshold,nms_threshold)
    # print(indices)
    for indice in indices:
        bbox = bboxes[indice]
        x,y,w,h = bbox[0],bbox[1],bbox[2],bbox[3]
        
        color_ndarray = COLORS[coco_classes.index(coco_classes[class_ids[indice]])]
        color_tuple = tuple([int(item) for id,item in enumerate(color_ndarray)])

        drawing(img,x,y,w,h,color_tuple,class_ids,indice,confidences)


def drawing(img,x,y,w,h,color_tuple,class_ids,indice,confidences):

    # draw a rectangle abound the objects
    cv.rectangle(img,(x,y),(x+w,y+h),color_tuple,thickness=2)
    
    font = cv.FONT_HERSHEY_SIMPLEX
    label = f'{coco_classes[class_ids[indice]].upper()} {int(confidences[indice]*100)}%'
    # get text size
    (text_width,text_heigh),baseline = cv.getTextSize(label,font,0.6,2) 
    # draw a rectangle above object rectangle
    cv.rectangle(img,(x,y),(x+text_width,y-text_heigh-baseline),color_tuple,thickness=-1)
    # put text above rectangle
    cv.putText(img, label ,(x,y-4),font,0.6, (0,0,0),thickness=2)
    

while cap.isOpened():
    ret, frame = cap.read()
    if not ret : break

    # construct a blob from the image
    blob = cv.dnn.blobFromImage(frame, scalefactor=1/255, size=(blob_size,blob_size),mean=(0,0,0)
                                 ,swapRB=True,crop=False)
             
    net.setInput(blob)
    
    # determine the output layer
    layers = net.getLayerNames()
    yolo_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]
    
    out_put = net.forward(yolo_layers)
    findObjects(out_put,frame)

    cv.imshow('Webcam',frame)
    if cv.waitKey(20) == 27: break

cv.destroyAllWindows()
cap.release()
