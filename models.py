import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
sys.path.append(r'C:\Users\13660\PycharmProjects\OFE-Reg\flownet2')
sys.path.append(r'C:\Users\13660\PycharmProjects\OFE-Reg\PWC\models')
sys.path.append(r'C:\Users\13660\PycharmProjects\OFE-Reg\RAFT\core')
from utils import crop_like
import flownet2.models as flownet2
import RAFT.core.raft as raft
import PWC.models.PWCNet as pwc
from torch.nn.init import kaiming_normal_, constant_
import time


# https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py 有batchnorm版本
def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def conv_3d(in_planes, out_planes, kernel_size, stride):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2),
        nn.ReLU(True)
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )


# flownetSimple
class flowNetS(nn.Module):
    def __init__(self, batchNorm=False):
        super(flowNetS, self).__init__()

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 2, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256, kernel_size=3, stride=1)
        self.conv4 = conv(self.batchNorm, 256, 512, kernel_size=3, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512, 3, 1)
        self.conv5 = conv(self.batchNorm, 512, 512, 3, 2)
        self.conv5_1 = conv(self.batchNorm, 512, 512, 3, 1)
        self.conv6 = conv(self.batchNorm, 512, 1024, 3, 2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024, 3, 1)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.deconv1 = deconv(194, 32)
        self.deconv0 = deconv(98, 16)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.predict_flow1 = predict_flow(98)
        self.predict_flow0 = predict_flow(20)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         kaiming_normal_(m.weight, 0.1)
        #         if m.bias is not None:
        #             constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         constant_(m.weight, 1)
        #         constant_(m.bias, 0)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv3_1 = self.conv3_1(out_conv3)
        out_conv4 = self.conv4(out_conv3_1)
        out_conv4_1 = self.conv4_1(out_conv4)
        out_conv5 = self.conv5(out_conv4_1)
        out_conv5_1 = self.conv5_1(out_conv5)
        out_conv6 = self.conv6(out_conv5_1)
        out_conv6_1 = self.conv6_1(out_conv6)

        flow6 = self.predict_flow6(out_conv6_1)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5_1)
        out_deconv5 = crop_like(self.deconv5(out_conv6_1), out_conv5_1)

        concat5 = torch.cat((out_conv5_1, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4_1)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4_1)

        concat4 = torch.cat((out_conv4_1, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3_1)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3_1)

        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        flow2_up = crop_like(self.upsampled_flow2_to_1(flow2), out_conv1)
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1)

        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        flow1 = self.predict_flow1(concat1)
        flow1_up = crop_like(self.upsampled_flow1_to_0(flow1), x)
        out_deconv0 = crop_like(self.deconv0(concat1), x)

        concat0 = torch.cat((x, out_deconv0, flow1_up), 1)
        flow0 = self.predict_flow0(concat0)

        if self.training:
            return flow0, flow1, flow2, flow3, flow4, flow5, flow6
        else:
            return flow0, flow1, flow2, flow3, flow4, flow5, flow6
            # return flow0


class affmodel(nn.Module):
    def __init__(self):
        super(affmodel, self).__init__()
        # 这里padding还有问题
        self.conv1 = conv_3d(2, 16, 7, (2, 2, 1))
        self.conv2 = conv_3d(16, 32, 5, (2, 2, 1))
        self.conv3 = conv_3d(32, 64, 3, 2)
        self.conv4 = conv_3d(64, 128, 3, 2)
        self.conv5 = conv_3d(128, 256, 3, 2)
        self.conv6 = conv_3d(256, 512, 3, 2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(8 * 8 * 512, 12)

    def forward(self, x):
        para = self.fc(self.flat(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x))))))))
        grid = F.affine_grid(para, x.size())
        out = F.grid_sample(x, grid)

        return out


# https://github.com/ily-R/Unsupervised-Optical-Flow/blob/master/models.py
def generate_grid(B, H, W, device):
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid = torch.transpose(grid, 1, 2)
    grid = torch.transpose(grid, 2, 3)
    grid = grid.to(device)
    return grid


# https://github.com/ily-R/Unsupervised-Optical-Flow/blob/master/models.py
class opticalFlowReg(nn.Module):
    def __init__(self, conv_predictor="flowNetS"):
        super(opticalFlowReg, self).__init__()
        #  加入不同OpticalFlow模型
        if "flownet2" in conv_predictor:
            parser = argparse.ArgumentParser()

            parser.add_argument('--start_epoch', type=int, default=1)
            parser.add_argument('--total_epochs', type=int, default=10000)
            parser.add_argument('--batch_size', '-b', type=int, default=4, help="Batch size")
            parser.add_argument('--train_n_batches', type=int, default=-1,
                                help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
            parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                                help="Spatial dimension to crop training samples for training")
            parser.add_argument('--gradient_clip', type=float, default=None)
            parser.add_argument('--schedule_lr_frequency', type=int, default=0,
                                help='in number of iterations (0 for no schedule)')
            parser.add_argument('--schedule_lr_fraction', type=float, default=10)
            parser.add_argument("--rgb_max", type=float, default=255.)

            parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
            parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
            parser.add_argument('--no_cuda', action='store_true')

            parser.add_argument('--seed', type=int, default=1)
            parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
            parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

            parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
            parser.add_argument('--validation_n_batches', type=int, default=-1)
            parser.add_argument('--render_validation', action='store_true',
                                help='run inference (save flows to file) and every validation_frequency epoch')

            parser.add_argument('--inference', action='store_true')
            parser.add_argument('--inference_visualize', action='store_true',
                                help="visualize the optical flow during inference")
            parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
                                help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
            parser.add_argument('--inference_batch_size', type=int, default=1)
            parser.add_argument('--inference_n_batches', type=int, default=-1)
            parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

            parser.add_argument('--resume', default='', type=str, metavar='PATH',
                                help='path to latest checkpoint (default: none)')
            parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

            parser.add_argument('--skip_training', action='store_true')
            parser.add_argument('--skip_validation', action='store_true')

            parser.add_argument('--fp16', action='store_true',
                                help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
            parser.add_argument('--fp16_scale', type=float, default=1024.,
                                help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
            args = parser.parse_args()
            self.predictor = flownet2.FlowNet2(args)

        elif "raft" in conv_predictor:
            parser = argparse.ArgumentParser()
            parser.add_argument('--name', default='raft', help="name your experiment")
            parser.add_argument('--stage', help="determines which dataset to use for training")
            parser.add_argument('--restore_ckpt', help="restore checkpoint")
            parser.add_argument('--small', action='store_true', help='use small model')
            parser.add_argument('--validation', type=str, nargs='+')

            parser.add_argument('--lr', type=float, default=0.00002)
            parser.add_argument('--num_steps', type=int, default=100000)
            parser.add_argument('--batch_size', type=int, default=6)
            parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
            parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
            parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

            parser.add_argument('--iters', type=int, default=12)
            parser.add_argument('--wdecay', type=float, default=.00005)
            parser.add_argument('--epsilon', type=float, default=1e-8)
            parser.add_argument('--clip', type=float, default=1.0)
            parser.add_argument('--dropout', type=float, default=0.0)
            parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
            parser.add_argument('--add_noise', action='store_true')
            args = parser.parse_args()
            self.predictor = raft.RAFT(args)

        elif "pwc" in conv_predictor:
            self.predictor = pwc.PWCDCNet()
        else:
            self.predictor = flowNetS()


    def stn(self, flow, frame):
        b, _, h, w = flow.shape
        frame = F.interpolate(frame, size=(h, w), mode='bilinear', align_corners=True)
        flow = torch.transpose(flow, 1, 2)
        flow = torch.transpose(flow, 2, 3)

        grid = flow + generate_grid(b, h, w, flow.device)

        factor = torch.FloatTensor([[[[2 / w, 2 / h]]]]).to(flow.device)
        grid = grid * factor - 1
        warped_frame = F.grid_sample(frame, grid, align_corners=True)  # 存疑

        return warped_frame

    def forward(self, x):

        flow_predictions = self.predictor(x)
        moving = x[:, 1, :, :]
        moving = torch.unsqueeze(moving, dim=1)
        warped_images = [self.stn(flow, moving) for flow in flow_predictions]
        #warped_images = self.stn(flow_predictions, moving)
        return flow_predictions, warped_images


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_test = opticalFlowReg(conv_predictor="pwc")
    model_test.to(device)
    test_batch = torch.rand(8, 2, 256, 256).to(device)
    output = model_test(test_batch)

    print(type(output))
