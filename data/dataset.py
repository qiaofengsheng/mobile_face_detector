from tools.anchor import *
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RetinaFaceDataset(Dataset):
    def __init__(self, txt_path, image_size):
        self.txt_path = txt_path
        self.image_size = image_size
        self.imgs_path, self.words = self.get_labels()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(0.5),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        # 打开图像
        image = cv2.imread(self.imgs_path[index])
        h, w, c = image.shape
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        ws, hs = w / self.image_size[1], h / self.image_size[0]
        labels = self.words[index]
        result_labels = []
        if len(labels) == 0:
            return image, [0, 0, 0, 0, 0]
        for idx, label in enumerate(labels):
            # bbox真实框的位置 x1,y1,x2,y2  lx,ly...
            x = label[0] / ws
            y = label[1] / hs
            w = label[2] / ws
            h = label[3] / hs
            result_labels.append([1, x + w / 2, y + h / 2, w, h])
        '''
        可添加自定义数据增强
        '''
        labels = generate_anchor_labels(result_labels)
        return self.transform(image), labels[0], labels[1], labels[2]

    def get_labels(self):
        imgs_path, words = [], []
        f = open(self.txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = self.txt_path.replace('label.txt', 'images/') + path.strip()
                imgs_path.append(path)
            else:
                line = line.strip().split(' ')
                label = [float(i) for i in line]
                labels.append(label)

        words.append(labels)
        return imgs_path, words


if __name__ == '__main__':
    data = RetinaFaceDataset(r'/data/face_det/data/widerface/train/label.txt', (640, 640))
    print(data[5])
