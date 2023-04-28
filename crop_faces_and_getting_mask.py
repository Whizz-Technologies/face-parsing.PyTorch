import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
from torchvision import transforms
from model import BiSeNet

class ImageSegmentation:
    def __init__(self):
        # Initialize face detection model
        self.mtcnn = MTCNN()
        # Initialize face recognition model
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        # Load segmentation model
        self.net = BiSeNet(n_classes=19)
        self.net.load_state_dict(torch.load(r'res\cp\79999_iter.pth', map_location=torch.device('cpu')))
        self.net.eval()
        # Set up data transformations
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # Set up image paths
        self.respth = r'./res/test_res'  # Directory to save output parsing maps
        os.makedirs(self.respth, exist_ok=True)
        self.orgpth=r'./res/cropped'
        os.makedirs(self.orgpth, exist_ok=True)
        # Set up class indices
        self.hair_class_index = 17
        self.neck_class_index = 14
    def segment_images(self, image_dir):
        image_list = os.listdir(image_dir)
        i = 0
        for image_name in image_list:
            image_path = os.path.join(image_dir, image_name)
            img = cv2.imread(image_path)
            # Detect faces in image
            boxes, _ = self.mtcnn.detect(img)
            for box in boxes:
                # Extract face region
                x1, y1 = int(box[1]), int(box[0])
                x2, y2 = int(box[3]), int(box[2])
                dx = int((x2 - x1) * 0.4)
                dy = int((y2 - y1) * 0.4)
                if(dx<40 and dy<40):
                    continue

                h,w=img.shape[:2]
                
                x1,y1,x2,y2=x1-dx,y1-dy,x2+dx,y2+dy
                x1=x1 if x1>0 else 0
                y1=y1 if y1>0 else 0
                x2=x2 if x2<h else h-1
                y2=y2 if y2<w else w-1
                cropped_img = img[x1:x2, y1:y2]
                
                resized_img = cv2.resize(cropped_img, (1024, 1024))
                # Save face region as image
                filename = f"{i}.jpg"
                filepath = os.path.join(self.orgpth, filename)
                cv2.imwrite(filepath, resized_img)
                
                # Segment hair and neck in face region
                image = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
                image = self.to_tensor(image)
                image = self.normalize(image)
                image = torch.unsqueeze(image, 0)
                with torch.no_grad():
                    out = self.net(image)
                    parsing = out[0].squeeze(0).cpu().numpy().argmax(0)
                hair_mask = (parsing == self.hair_class_index).astype('uint8') * 255
                neck_mask = (parsing == self.neck_class_index).astype('uint8') * 255
                mask = cv2.bitwise_or(hair_mask, neck_mask)
                mask_path = os.path.join(self.respth, f'{i}_mask.png')
                cv2.imwrite(mask_path, mask)
                i += 1
        print('Results saved at:', self.respth,' and ',self.orgpth)
#create an object first for ImageSegmentation()
#Then call the method object.segment_images(path of images directory)
