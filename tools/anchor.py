import math
import numpy as np
from data.config import *
from tools.utils import *

anchors = {
    cfg_mobilenet['image_size'] // cfg_mobilenet['steps'][2]: [[256, 256], [512, 512]],
    cfg_mobilenet['image_size'] // cfg_mobilenet['steps'][1]: [[64, 64], [128, 128]],
    cfg_mobilenet['image_size'] // cfg_mobilenet['steps'][0]: [[16, 16], [32, 32]],
}
anchor_areas = {
    cfg_mobilenet['image_size'] // cfg_mobilenet['steps'][2]: [w * h for w, h in anchors[
        cfg_mobilenet['image_size'] // cfg_mobilenet['steps'][2]]],
    cfg_mobilenet['image_size'] // cfg_mobilenet['steps'][1]: [w * h for w, h in anchors[
        cfg_mobilenet['image_size'] // cfg_mobilenet['steps'][1]]],
    cfg_mobilenet['image_size'] // cfg_mobilenet['steps'][0]: [w * h for w, h in anchors[
        cfg_mobilenet['image_size'] // cfg_mobilenet['steps'][0]]]
}


def generate_anchor_labels(boxes):
    input_size = cfg_mobilenet['image_size']
    labels = {}
    for feature_size, _anchor in anchors.items():
        labels[feature_size] = np.zeros((feature_size, feature_size, 2, 5))
        for box in boxes:
            target_cls, target_x, target_y, target_w, target_h = box
            cx_offset, center_x = math.modf(target_x * feature_size / input_size)
            cy_offset, center_y = math.modf(target_y * feature_size / input_size)
            for i, anchor in enumerate(_anchor):
                # anchor_x = center_x * input_size / feature_size
                # anchor_y = center_y * input_size / feature_size
                anchor_w = anchor[0]
                anchor_h = anchor[1]
                h_offset = np.log(target_h / anchor_h)
                w_offset = np.log(target_w / anchor_w)
                # iou_score = iou(torch.Tensor([target_x, target_y, target_w, target_h]),
                #                 torch.Tensor([[anchor_x, anchor_y, anchor_w, anchor_h]]))
                anchor_area=anchor_h*anchor_w
                target_area = target_w*target_h
                iou_score = min(anchor_area,target_area)/max(anchor_area,target_area)
                if iou_score >= cfg_mobilenet['positive_thresh']:
                    labels[feature_size][int(center_y), int(center_x), i] = np.array(
                        [1, cx_offset, cy_offset, w_offset, h_offset])
    return labels[80], labels[40], labels[20]


if __name__ == '__main__':
    boxes = [[1, 110, 120, 300, 400], [1, 200, 200, 500, 500]]
    c = generate_anchor_labels(boxes)
    print(c[0].shape)
    print(c[1].shape)
    print(c[2].shape)
