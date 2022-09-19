import torch


def iou(box, boxes):
    '''
    :param box: x,y,w,h
    :param boxes: x,y,w,h
    :return:
    '''
    area1 = box[2] * box[3]
    area2 = boxes[:, 2] * boxes[:, 3]
    x11 = box[0] - box[2] / 2
    y11 = box[1] - box[3] / 2
    x12 = box[0] + box[2] / 2
    y12 = box[1] + box[3] / 2
    x21 = boxes[:, 0] - boxes[:, 2] / 2
    y21 = boxes[:, 1] - boxes[:, 3] / 2
    x22 = boxes[:, 0] + boxes[:, 2] / 2
    y22 = boxes[:, 1] + boxes[:, 3] / 2

    x1 = torch.maximum(x11, x21)
    y1 = torch.maximum(y11, y21)
    x2 = torch.minimum(x12, x22)
    y2 = torch.minimum(y12, y22)
    w = torch.clamp((x2 - x1), min=0)
    h = torch.clamp((y2 - y1), min=0)
    inter = w * h
    return inter / (area1 + area2 - inter)


if __name__ == '__main__':
    box = torch.tensor([1, 2, 3, 4])
    boxes = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]])
    print(iou(box, boxes))
