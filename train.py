import torch
import tqdm
from torch import optim
from data.dataset import *
from model.mobile_face import MobileFace
from tools.loss import *
from tools.anchor import *


class Train:
    def __init__(self,weight_path=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.net = MobileFace(cfg_mobilenet)
        if weight_path is not None:
            self.net.load_state_dict(torch.load(weight_path))
            print('successfully load weight!')

        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=0.00005)
        self.train_dataset = RetinaFaceDataset(r"/data/face_det/data/widerface/train/label.txt", (640, 640))
        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg_mobilenet['batch_size'], shuffle=True,
                                       num_workers=0)
        self.net.to(self.device)

    def train(self):
        for epoch in range(cfg_mobilenet['epoch']):
            with tqdm.tqdm(self.train_loader) as t1:
                for i, (images, targets1, targets2, targets3) in enumerate(self.train_loader):
                    images, targets1, targets2, targets3 = images.to(self.device), targets1.to(
                        self.device), targets2.to(self.device), targets3.to(self.device)
                    predicts1, predicts2, predicts3 = self.net(images)
                    self.optimizer.zero_grad()
                    loss1 = loss_function(predicts1.float(), targets1.float(), 2)
                    loss2 = loss_function(predicts2.float(), targets2.float(), 2)
                    loss3 = loss_function(predicts3.float(), targets3.float(), 2)
                    loss = loss1+loss2+loss3
                    # loss= loss.float()
                    t1.set_postfix(epoch=epoch, loss=loss.item())
                    t1.update(1)
                    loss.backward()
                    self.optimizer.step()
            torch.save(self.net.state_dict(), '/home/situ/qfs/temp/c/mobile_face_detector/checkpoints/mobile_face_det_balance.pth')
            print('save model successfully!')


if __name__ == '__main__':
    Train("/home/situ/qfs/temp/c/mobile_face_detector/checkpoints/mobile_face_det.pth").train()
