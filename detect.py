import os
import time

import cv2
import numpy as np
import torch
import tqdm
from charset_normalizer import detect

from tools.nms import *
from model.mobile_face import *
from tools.anchor import *
from cv2 import getTickCount, getTickFrequency

class Detector:
    def __init__(self, weights_path, threshold=0.5):
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.net = MobileFace(cfg_mobilenet).to(self.device)
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
        anchor = torch.Tensor(anchor).to(self.device)
        anchor_index = idx[:, 3]
        center_x = (idx[:, 2] + vec[:, 1]) * step / scale_w
        center_y = (idx[:, 1] + vec[:, 2]) * step / scale_h
        w = torch.exp(vec[:, 3].float()) * anchor[anchor_index, 0] / scale_w
        h = torch.exp(vec[:, 4].float()) * anchor[anchor_index, 1] / scale_h
        score = torch.sigmoid(vec[:, 0])
        return torch.stack([center_x, center_y, w, h, score], dim=1)

    def detect_image(self, image):
        h, w, c = image.shape
        image = cv2.resize(image, (640, 640))
        image = image / 255.0
        scale_h = 640 / h
        scale_w = 640 / w
        image = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        output0, output1, output2 = self.net(image)
        idx80, vec80 = self._filter(output0, self.threshold)
        idx40, vec40 = self._filter(output1, self.threshold)
        idx20, vec20 = self._filter(output2, self.threshold)
        result80 = self._parse(idx80, vec80, 8, anchors[80], scale_h, scale_w)
        result40 = self._parse(idx40, vec40, 16, anchors[40], scale_h, scale_w)
        result20 = self._parse(idx20, vec20, 32, anchors[20], scale_h, scale_w)
        result = result80.tolist()+result40.tolist()+result20.tolist()
        bboxes = np.array(result)
        if len(result)!=0:
            idx = py_cpu_nms(bboxes, 0.1)
            bboxes = bboxes[idx]
        return bboxes

    def detect_video(self,video_path):
        cap = cv2.VideoCapture(video_path)
        _, frame = cap.read()
        while _:
            _, frame = cap.read()
            bboxes = self.detect_image(frame)
            for box in bboxes:
                x1 = int(box[0] - box[2] / 2)
                y1 = int(box[1] - box[3] / 2)
                x2 = int(box[0] + box[2] / 2)
                y2 = int(box[1] + box[3] / 2)
                score = box[4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, 'face:' + str(round(score, 3)), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
            cv2.imshow('frame',frame)
            if cv2.waitKey(10)&ord('q')==0XFF:
                break



if __name__ == '__main__':
    weight_path = 'checkpoints/mobile_face_det_balance.pth'
    image_path = '/data/face_det/data/widerface/val/images/41--Swimming'
    detector = Detector(weight_path, 0.8)
    detector.detect_video('/data/speak/bilibili/speak/2.mp4')
    # time_list=[]
    # for i in tqdm.tqdm(os.listdir(image_path)):
    #     path = os.path.join(image_path,i)
    #     st = time.time()
    #     detector.detect(path)
    #     spend_time = time.time()-st
    #     time_list.append(spend_time)
    #
    # print('avg time:',np.mean(time_list))
    # print('min time:',np.min(time_list))
    # print('max time:',np.max(time_list))
    # print('avg fps:',1/np.mean(time_list))