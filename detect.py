import cv2
import torch
from tools.nms import *
from model.mobile_face import *
from tools.anchor import *


class Detector:
    def __init__(self, weights_path, threshold=0.5):
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.net = MobileFace(cfg_mobilenet)
        self.net.load_state_dict(torch.load(weights_path))
        self.net.eval()

    def _filter(self, output, threshold):
        output = output.reshape(output.shape[0], output.shape[1], output.shape[2], 2, -1)
        score = torch.sigmoid(output[..., 0])
        obj_mask = score >= threshold
        idx = obj_mask.nonzero()
        vec = output[obj_mask]
        return idx, vec

    def _parse(self, idx, vec, step, anchor, scale_h, scale_w):
        anchor = torch.Tensor(anchor)
        anchor_index = idx[:, 3]
        center_x = (idx[:, 2] + vec[:, 1]) * step / scale_w
        center_y = (idx[:, 1] + vec[:, 2]) * step / scale_h
        w = torch.exp(vec[:, 3].float()) * anchor[anchor_index, 0] / scale_w
        h = torch.exp(vec[:, 4].float()) * anchor[anchor_index, 1] / scale_h
        score = torch.sigmoid(vec[:, 0])
        return torch.stack([center_x, center_y, w, h, score], dim=1)

    def detect(self, image_path):
        image = cv2.imread(image_path)
        image_ = image.copy()
        h, w, c = image.shape
        image = cv2.resize(image, (640, 640))
        image = image / 255.0
        scale_h = 640 / h
        scale_w = 640 / w
        image = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
        output0, output1, output2 = self.net(image)
        idx80, vec80 = self._filter(output0, self.threshold)
        idx40, vec40 = self._filter(output1, self.threshold)
        idx20, vec20 = self._filter(output2, self.threshold)
        result80 = self._parse(idx80, vec80, 8, anchors[80], scale_h, scale_w)
        result40 = self._parse(idx40, vec40, 16, anchors[40], scale_h, scale_w)
        result20 = self._parse(idx20, vec20, 32, anchors[20], scale_h, scale_w)
        result = []
        if result20.shape[0] != 0:
            result.append(result20)
        if result40.shape[0] != 0:
            result.append(result40)
        if result80.shape[0] != 0:
            result.append(result80)
        bboxes = torch.cat(result, dim=0).detach().numpy()
        idx = py_cpu_nms(bboxes, 0.1)
        bboxes = bboxes[idx]

        for box in bboxes:
            print(box)
            x1 = int(box[0] - box[2] / 2)
            y1 = int(box[1] - box[3] / 2)
            x2 = int(box[0] + box[2] / 2)
            y2 = int(box[1] + box[3] / 2)
            cv2.rectangle(image_, (x1, y1), (x2, y2), (0, 0, 255))
        cv2.imshow('w', image_)
        cv2.waitKey(0)


if __name__ == '__main__':
    weight_path = 'checkpoints/mobile_face_det_balance.pth'
    Detector(weight_path, 0.5).detect("/home/situ/图片/R1.jpeg")
