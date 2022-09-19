import numpy as np


def py_cpu_nms(dets, thresh):
    '''
    MNS
    :param det: N * 4
    :param thresh:
    :return:
    '''
    x1 = dets[:, 0]-dets[:,2]/2
    y1 = dets[:, 1]-dets[:,3]/2
    x2 = dets[:, 0]+dets[:,2]/2
    y2 = dets[:, 1]+dets[:,3]/2
    score = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(score)[::-1]
    keep = []
    while order.size > 0:
        index = order[0]
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        w = np.maximum(0,xx2 - xx1 + 1)
        h = np.maximum(0,yy2 - yy1 + 1)
        inter = w*h
        overlap = inter / (areas[index]+areas[order[1:]]-inter)
        indx = np.where(overlap<=thresh)[0]
        order = order[indx + 1]
    return keep


if __name__ == '__main__':
    boxes = np.array([[100, 100, 210, 210, 0.72],
                      [250, 250, 420, 420, 0.8],
                      [220, 220, 320, 330, 0.92],
                      [100, 100, 210, 210, 0.72],
                      [230, 240, 325, 330, 0.81],
                      [220, 230, 315, 340, 0.9]])


    c=py_cpu_nms(boxes,0.3)
    print(boxes[c])