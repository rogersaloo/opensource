import torch
from super_gradients.training import models

DEVICE = 'cuda' if torch.cuda.is_available() else cpu
MODEL_ARCH = 'yolo_nas_l'

model = models.get(MODEL_ARCH, pretrained_weights="coco").to(DEVICE)