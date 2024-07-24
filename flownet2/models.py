import torch
import torch.nn as nn
from torch.nn import init
import argparse

import math
import numpy as np

try:
    from networks.resample2d_package.resample2d import Resample2d
    from networks.channelnorm_package.channelnorm import ChannelNorm

    from networks import FlowNetC
    from networks import FlowNetS
    from networks import FlowNetSD
    from networks import FlowNetFusion

    from networks.submodules import *
except:
    from .networks.resample2d_package.resample2d import Resample2d
    from .networks.channelnorm_package.channelnorm import ChannelNorm

    from .networks import FlowNetC
    from .networks import FlowNetS
    from .networks import FlowNetSD
    from .networks import FlowNetFusion

    from .networks.submodules import *
'Parameter count = 162,518,834'

class FlowNet2(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow = 20.):
        super(FlowNet2,self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        if args.fp16:
            self.resample1 = nn.Sequential(
                            tofp32(), 
                            Resample2d(),
                            tofp16()) 
        else:
            self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample2 = nn.Sequential(
                            tofp32(), 
                            Resample2d(),
                            tofp16()) 
        else:
            self.resample2 = Resample2d()


        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)

        # Block (FlowNetSD)
        self.flownets_d = FlowNetSD.FlowNetSD(args, batchNorm=self.batchNorm) 
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest') 
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest') 

        if args.fp16:
            self.resample3 = nn.Sequential(
                            tofp32(), 
                            Resample2d(),
                            tofp16()) 
        else:
            self.resample3 = Resample2d()

        if args.fp16:
            self.resample4 = nn.Sequential(
                            tofp32(), 
                            Resample2d(),
                            tofp16()) 
        else:
            self.resample4 = Resample2d()

        # Block (FLowNetFusion)
        self.flownetfusion = FlowNetFusion.FlowNetFusion(args, batchNorm=self.batchNorm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)

    def init_deconv_bilinear(self, weight):
        f_shape = weight.size()
        heigh, width = f_shape[-2], f_shape[-1]
        f = np.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([heigh, width])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        min_dim = min(f_shape[0], f_shape[1])
        weight.data.fill_(0.)
        for i in range(min_dim):
            weight.data[i,i,:,:] = torch.from_numpy(bilinear)
        return 

    def forward(self, inputs):
        # rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        # x = (inputs - rgb_mean) / self.rgb_max
        # x1 = x[:,:,0,:,:]
        # x2 = x[:,:,1,:,:]
        # x = torch.cat((x1,x2), dim = 1)

        x1 = inputs[:,0,:,:]
        x2 = inputs[:,1,:,:]
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)

        # flownetc
        flownetc_flow2 = self.flownetc(inputs)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)
        
        # warp img1 to img0; magnitude of diff between img0 and and warped_img1, 
        resampled_img1 = self.resample1(x2, flownetc_flow)
        diff_img0 = x1 - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ; 
        concat1 = torch.cat((inputs, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)
        
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow) 

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x2, flownets1_flow)
        diff_img0 = x1 - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat((inputs, resampled_img1, flownets1_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)

        diff_flownets2_flow = self.resample4(x2, flownets2_flow)
        # if not diff_flownets2_flow.volatile:
        #     diff_flownets2_flow.register_hook(save_grad(self.args.grads, 'diff_flownets2_flow'))

        diff_flownets2_img1 = self.channelnorm((x1-diff_flownets2_flow))
        # if not diff_flownets2_img1.volatile:
        #     diff_flownets2_img1.register_hook(save_grad(self.args.grads, 'diff_flownets2_img1'))

        # flownetsd
        flownetsd_flow2 = self.flownets_d(inputs)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)
        
        diff_flownetsd_flow = self.resample3(x2, flownetsd_flow)
        # if not diff_flownetsd_flow.volatile:
        #     diff_flownetsd_flow.register_hook(save_grad(self.args.grads, 'diff_flownetsd_flow'))

        diff_flownetsd_img1 = self.channelnorm((x1-diff_flownetsd_flow))
        # if not diff_flownetsd_img1.volatile:
        #     diff_flownetsd_img1.register_hook(save_grad(self.args.grads, 'diff_flownetsd_img1'))

        # concat img0 flownetsd, flownets2, norm_flownetsd, norm_flownets2, diff_flownetsd_img1, diff_flownets2_img1
        concat3 = torch.cat((x1, flownetsd_flow, flownets2_flow, norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1, diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)

        # if not flownetfusion_flow.volatile:
        #     flownetfusion_flow.register_hook(save_grad(self.args.grads, 'flownetfusion_flow'))

        return flownetfusion_flow, flownetfusion_flow

class FlowNet2C(FlowNetC.FlowNetC):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2C,self).__init__(args, batchNorm=batchNorm, div_flow=20)
        self.rgb_max = args.rgb_max

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]

        # FlownetC top input stream
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)
        
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b) # False
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)

        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)

        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1,out_deconv3,flow4_up),1)

        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up),1)

        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)

class FlowNet2S(FlowNetS.FlowNetS):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2S,self).__init__(args, input_channels = 6, batchNorm=batchNorm)
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow
        
    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)

class FlowNet2SD(FlowNetSD.FlowNetSD):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2SD,self).__init__(args, batchNorm=batchNorm)
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5       = self.predict_flow5(out_interconv5)

        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4       = self.predict_flow4(out_interconv4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3       = self.predict_flow3(out_interconv3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)

class FlowNet2CS(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow = 20.):
        super(FlowNet2CS,self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        if args.fp16:
            self.resample1 = nn.Sequential(
                            tofp32(), 
                            Resample2d(),
                            tofp16()) 
        else:
            self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)
        
        # warp img1 to img0; magnitude of diff between img0 and and warped_img1, 
        resampled_img1 = self.resample1(x[:,3:,:,:], flownetc_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1 
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ; 
        concat1 = torch.cat((x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)
        
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow) 

        return flownets1_flow

class FlowNet2CSS(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow = 20.):
        super(FlowNet2CSS,self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args

        self.channelnorm = ChannelNorm()

        # First Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

        if args.fp16:
            self.resample1 = nn.Sequential(
                            tofp32(), 
                            Resample2d(),
                            tofp16()) 
        else:
            self.resample1 = Resample2d()

        # Block (FlowNetS1)
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample2 = nn.Sequential(
                            tofp32(), 
                            Resample2d(),
                            tofp16()) 
        else:
            self.resample2 = Resample2d()


        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest') 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)
        
        # warp img1 to img0; magnitude of diff between img0 and and warped_img1, 
        resampled_img1 = self.resample1(x[:,3:,:,:], flownetc_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1 
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag ; 
        concat1 = torch.cat((x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)
        
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow) 

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = self.resample2(x[:,3:,:,:], flownets1_flow)
        diff_img0 = x[:,:3,:,:] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat((x, resampled_img1, flownets1_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)

        return flownets2_flow

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024.,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    args = parser.parse_args()

    flownet2 = FlowNet2(args)
    flownet2.to(device)
    print(flownet2)
    test_batch = torch.rand(4, 2, 256, 256).to(device)
    output = flownet2(test_batch)
    print(len(output))
    print(output[0].size())
    #correlation.corr()