import numpy as np
import cv2
import os
from django.conf import settings

from PIL import Image
import torch
from torchvision.transforms import functional as F
from .model.unet.utils import load_seunet
from .constants import TOOTH_COLORMAP


class SegmentationPredictor:
    def __init__(self, model, mean=0.458, std=0.173, cuda=True) -> None:
        self.model = model
        self.model.eval()
        self.cuda = cuda
        self.mean = mean
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
        predictions = F.resize(predictions, origin_shape,
                               F.InterpolationMode.NEAREST)
        return predictions.squeeze(0)


def label_mask_to_bbox(mask: np.ndarray):
    bbox_dict = {}
    area_threshold = 1000
    for label in np.unique(mask):
        if label == 0:
            continue
        label_mask = mask == label
        label_mask = label_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(
            contours, key=lambda x: cv2.contourArea(x), reverse=True)
        if len(contours) == 0:
            continue
        bbox = cv2.boundingRect(contours[0])
        if ((bbox[2] * bbox[3]) > area_threshold):
            bbox_dict[label] = bbox
    return bbox_dict


@torch.no_grad()
def main(file_bytes):

    if file_bytes is None or len(file_bytes) == 0:
        raise ValueError("file_bytes vazio")

    cuda = False
    MODEL_PATH = os.path.join(settings.BASE_DIR, "imagemproc", "SEUnet32.pth")
    model = load_seunet(MODEL_PATH, 33, cuda=cuda)

    predictor = SegmentationPredictor(model, cuda=cuda)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Deu xabu ao decodificar a imagem")
    image_bgr_arr = image[:, :, ::-1]
    image_gray_arr = cv2.cvtColor(image_bgr_arr, cv2.COLOR_BGR2GRAY)

    enumeration_prediction_mask = predictor.predict(image_gray_arr)
    enumeration_prediction_mask = enumeration_prediction_mask.cpu().numpy()
    enumeration_prediction_bbox_dict = label_mask_to_bbox(
        enumeration_prediction_mask)

    thickness = 2

    for enumeration_id, enumeration_bbox in enumeration_prediction_bbox_dict.items():
        color = TOOTH_COLORMAP[enumeration_id - 1]["color"]
        name = TOOTH_COLORMAP[enumeration_id - 1]["name"]
        image = cv2.rectangle(
            image,
            (enumeration_bbox[0], enumeration_bbox[1]),
            (enumeration_bbox[0] + enumeration_bbox[2],
             enumeration_bbox[1] + enumeration_bbox[3]),
            color,
            thickness,
        )
        image = cv2.putText(
            image,
            str(name),
            (enumeration_bbox[0], enumeration_bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return image
