import torch
from model.layer_utils import *
import torchvision.models._utils as _utils
from data.config import *


class RetinaFace(nn.Module):
    def __init__(self, config, is_train=True):
        super(RetinaFace, self).__init__()
        self.config = config
        self.is_train = is_train
        backbone = MobileNetV1(config['model_size'])

        self.body = _utils.IntermediateLayerGetter(backbone, config['stage_layers'])
        stage_input_channels = config['input_channels']
        stage_input_channel_list = [
            stage_input_channels * 2,
            stage_input_channels * 4,
            stage_input_channels * 8,
        ]
        output_channels = config['output_channels']
        self.fpn = FPN(stage_input_channel_list, output_channels)
        self.ssh1 = SSH(output_channels, output_channels)
        self.ssh2 = SSH(output_channels, output_channels)
        self.ssh3 = SSH(output_channels, output_channels)

        self.head = DetectHead(output_channels)


    def forward(self, x):
        stage1 = self.body['stage1'](x)
        stage2 = self.body['stage2'](stage1)
        stage3 = self.body['stage3'](stage2)
        fpn_stages = self.fpn([stage1, stage2, stage3])

        # SSH
        feature1 = self.ssh1(fpn_stages[0])
        feature2 = self.ssh1(fpn_stages[1])
        feature3 = self.ssh1(fpn_stages[2])

        # Head
        output1 = self.head(feature1).permute(0,2,3,1)
        output2 = self.head(feature2).permute(0,2,3,1)
        output3 = self.head(feature3).permute(0,2,3,1)

        output = (output1,output2,output3)

        return output


if __name__ == '__main__':
    x = torch.randn(1, 3, 640, 640)
    net = RetinaFace(cfg_mobilenet)
    for i in net(x):
        c=torch.argmax(i[...,-2:],dim=3)
        print(c)
        print(c.shape)
