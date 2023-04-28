import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from model import BiSeNet
def evaluate():
    # Set up model
    net = BiSeNet(n_classes=19)
    net.load_state_dict(torch.load('res/cp/79999_iter.pth', map_location=torch.device('cpu')))
    net.eval()
    # Set up data transformations
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # Set up image paths
    dspth = r'C:\Users\desir\OneDrive\Documents\computer vision\open_images\images'  # Directory containing input images
    respth = r'./res/test_res'  # Directory to save output parsing maps
    os.makedirs(respth, exist_ok=True)
    image_list = os.listdir(dspth)
    hair_class_index = 17  # Update with the correct class index for hair in your case
    neck_class_index = 14  # Update with the correct class index for shirt in your case
    for image_name in image_list:
        image_path = os.path.join(dspth, image_name)
        image = Image.open(image_path).convert('RGB')
        image = to_tensor(image)
        image = normalize(image)
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            out = net(image)
            parsing = out[0].squeeze(0).cpu().numpy().argmax(0)
        # Extract hair mask based on specified class index
        hair_mask = (parsing == hair_class_index).astype('uint8') * 255
        # Extract shirt mask based on specified class index
        neck_mask = (parsing == neck_class_index).astype('uint8') * 255
        # Combine the hair and shirt masks
        mask = cv2.bitwise_or(hair_mask, neck_mask)
        # Save the combined mask as an image
        mask_path = os.path.join(respth, f'{image_name[:-4]}_mask.png')
        cv2.imwrite(mask_path, mask)
    print('Results saved at:', respth)
if __name__ == '__main__':
    evaluate()
