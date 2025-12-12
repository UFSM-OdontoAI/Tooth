import numpy as np
import cv2
import os
from django.conf import settings

from PIL import Image
import json
import torch
from torchvision.transforms import functional as F
from .model.unet.utils import load_unet, load_seunet


tooth_colormap = {
    0: dict(name= 11, color=(173, 216, 230)),
    1: dict(name= 12, color=(0, 191, 255)),
    2: dict(name= 13, color=(30, 144, 255)),
    3: dict(name= 14, color=(0,   0, 255)),
    4: dict(name= 15, color= (0,   0, 139)),
    5: dict(name= 16, color=(72,  61, 139)),
    6: dict(name= 17, color=(123, 104, 238)),
    7: dict(name= 18, color=(138,  43, 226)),
    8: dict(name= 21, color=(128,   0, 128)),
    9: dict(name= 22, color=(218, 112, 214)),
    10: dict(name= 23, color=(255,   0, 255)),
    11: dict(name= 24, color=(255,  20, 147)),
    12: dict(name= 25, color=(176,  48,  96)),
    13: dict(name= 26, color=(220,  20,  60)),
    14: dict(name= 27, color=(240, 128, 128)),
    15: dict(name= 28, color=(255,  69,   0)),
    16: dict(name= 31, color=(255, 165,   0)),
    17: dict(name= 32, color=(244, 164,  96)),
    18: dict(name= 33, color=(240, 230, 140)),
    19: dict(name= 34, color=(128, 128,   0)),
    20: dict(name= 35, color=(139,  69,  19)),
    21: dict(name= 36, color=(255, 255,   0)),
    22: dict(name= 37, color=(154, 205,  50)),
    23: dict(name= 38, color=(124, 252,   0)),
    24: dict(name= 41, color=(144, 238, 144)),
    25: dict(name= 42, color=(143, 188, 143)),
    26: dict(name= 43, color=( 34, 139,  34)),
    27: dict(name= 44, color=(  0, 255, 127)),
    28: dict(name= 45, color=(  0, 255, 255)),
    29: dict(name= 46, color=(  0, 139, 139)),
    30: dict(name= 47, color=(128, 128, 128)),
    31: dict(name= 48, color=(255, 255, 255)),
}


class SegmentationPredictor:
    def __init__(self, model, mean=0.458, std=0.173, cuda=True) -> None:
        self.model = model
        self.model.eval()
        self.cuda = cuda
        self.mean = mean  # default mean and std here are caculated from dentex dataset
        self.std = std

    def predict(self, image: np.ndarray) -> torch.Tensor:
        origin_shape = image.shape
        image = Image.fromarray(image).resize((256, 256))
        image = F.to_tensor(image)
        image = F.normalize(image, [self.mean], [self.std])
        if self.cuda:
            image = image.cuda()
        image = image.unsqueeze(0)

        predictions = self.model(image)
        predictions = predictions.squeeze(0)
        predictions = torch.argmax(predictions, dim=0, keepdim=True)
        predictions = F.resize(predictions, origin_shape, F.InterpolationMode.NEAREST)
        return predictions.squeeze(0)

def label_mask_to_bbox(mask: np.ndarray):
    """
    convert segmentation mask to bbox, only keep the largest connected component for each label
    mask: (H, W)
    return: bbox_dict, key is label, value is bbox (x, y, w, h)
    """
    bbox_dict = {}
    area_threshold = 1000
    for label in np.unique(mask):
        if label == 0:
            continue
        label_mask = mask == label
        label_mask = label_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        if len(contours) == 0:
            continue
        bbox = cv2.boundingRect(contours[0])
        if ((bbox[2] * bbox[3]) > area_threshold):
    #        print (f"label={tooth_colormap[label]["name"]}, area = {bbox[2]*bbox[3]}")
            bbox_dict[label] = bbox
    return bbox_dict

@torch.no_grad()
def main(file_path):
    cuda = False#torch.cuda.is_available() 

    #model = load_unet("output_unet_enum9_08-11_07-27/last_epoch.pth", 33, cuda=cuda)
    MODEL_PATH = os.path.join(settings.BASE_DIR, "imagemproc", "SEUnet32.pth")
#    model = load_seunet("SEUnet32.pth", 33, cuda=cuda)
    model = load_seunet(MODEL_PATH, 33, cuda=cuda)

    predictor = SegmentationPredictor(model, cuda=cuda)

    file_bytes = np.frombuffer(file_path.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#    image = cv2.imread(file_path)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    image_bgr_arr = image[:, :, ::-1]
    image_gray_arr = cv2.cvtColor(image_bgr_arr, cv2.COLOR_BGR2GRAY)

    enumeration_prediction_mask = predictor.predict(image_gray_arr)
    enumeration_prediction_mask = enumeration_prediction_mask.cpu().numpy()
    enumeration_prediction_bbox_dict = label_mask_to_bbox(enumeration_prediction_mask)

    # Line thickness of 2 px
    thickness = 2

    for enumeration_id, enumeration_bbox in enumeration_prediction_bbox_dict.items():
#        enumeration_bboxes_each_seg_model[-1].update({enumeration_id: enumeration_bbox})
#        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        image = cv2.rectangle(image, (enumeration_bbox[0], enumeration_bbox[1]), (enumeration_bbox[0]+enumeration_bbox[2], enumeration_bbox[1]+enumeration_bbox[3]), tooth_colormap[enumeration_id-1]["color"], thickness)
        image = cv2.putText(image, str(tooth_colormap[enumeration_id-1]["name"]), (enumeration_bbox[0], enumeration_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, tooth_colormap[enumeration_id-1]["color"], thickness, cv2.LINE_AA)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return image

if __name__ == "__main__":
    outimage = main('train_66.png')
    cv2.imwrite('segmented.png',outimage)
    
