from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import os
mtcnn = MTCNN()
# If required, create a face detection pipeline using MTCNN:
model = InceptionResnetV1(pretrained='vggface2').eval()
img=cv2.imread(r"C:\Users\desir\OneDrive\Documents\computer vision\open_images\images\test.jpg")
boxes, _ = mtcnn.detect(img)
i = 0
for box in boxes:
    x1, y1 = int(box[1]), int(box[0])
    x2, y2 = int(box[3]), int(box[2])
    dx = int(abs(x2 - x1) * 0.4)
    dy = int(abs(y2 - y1) * 0.4)
    print(dx,dy)
    if(dx<40 and dy<40):
        continue
    filename = f"{i}.jpg"
    filepath = os.path.join(r"C:\Users\desir\Downloads", filename)
    h,w=img.shape[:2]
    print(x1,y1,x2,y2,h,w)
    x1,y1,x2,y2=x1-dx,y1-dy,x2+dx,y2+dy
    
    x1=x1 if x1>0 else 0
    y1=y1 if y1>0 else 0
    x2=x2 if x2<h else h-1
    y2=y2 if y2<w else w-1
    print(x1,y1,x2,y2,h,w)
    cropped_img = img[x1:x2, y1:y2]
    
    resized_img = cv2.resize(cropped_img, (1024, 1024))
    
    cv2.imwrite(filepath, resized_img)
    
    i += 1
