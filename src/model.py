import json
import io, base64
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection.retinanet import retinanet_resnet50_fpn_v2


class RetinaNet:
    def __init__(self):
        self.model = retinanet_resnet50_fpn_v2(weights='COCO_V1')
        self.model.eval()
        self.inference_transform = transforms.Compose([transforms.ToTensor()])
        with open('./coco_labels.json', 'r') as f:
            self.index2labels = json.load(f)
        self.threshold = 0.5
    
    @torch.no_grad()
    def __call__(self, img):
        img = self.inference_transform(img).unsqueeze(0)
        preds = self.model(img)[0]
        idx_list = []

        for idx, score in enumerate(preds['scores']) :
            if score > self.threshold:
                idx_list.append(idx)

        preds['boxes'] = preds['boxes'][idx_list].numpy().tolist()
        preds['labels'] = [self.index2labels[str(x)] for x in preds['labels'][idx_list].numpy().tolist()]
        preds['scores'] = preds['scores'][idx_list].numpy().tolist()

        return preds