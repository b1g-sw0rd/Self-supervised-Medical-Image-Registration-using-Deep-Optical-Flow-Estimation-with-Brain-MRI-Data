import argparse
import os
import torch
from models import opticalFlowReg
from utils import averageMeter, dice_average,seed_everything, dist_hausdorff, MSE, PSNR, MI,CORR, flow_mag
from loss import OFEloss
import time
from dataset import eval_random_ds,real_ds
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity
import ants
import torchvision.transforms as transforms


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 调用gpu计算如果可用，否则cpu
device = torch.device('cuda')
# 打印间隔
PRINT_INTERVAL = 2


def epoch(model, data, criterion):
    model.eval()
    avg_dice = averageMeter()
    avg_mse = averageMeter()
    avg_psnr = averageMeter()
    avg_hausdorff = averageMeter()
    avg_ssim_img = averageMeter()
    avg_ssim_seg = averageMeter()
    avg_mi = averageMeter()
    avg_corr = averageMeter()
    avg_loss = averageMeter()
    avg_batch_time = averageMeter()
    avg_smooth_loss = averageMeter()
    avg_photo_loss = averageMeter()
    avg_corr_loss = averageMeter()
    avg_mag = averageMeter()
    length = eval_length

    i = 0
    tic = time.time()

    for imgs in data:  # imgs 包含前1通道fixed 后1 moving
        # imgs = imgs.to(device)
        segs = imgs["seg_c"].as_tensor().to(device)
        imgs = imgs["image_c"].as_tensor().to(device)
        imgs = imgs.float()
        fixed_img = torch.unsqueeze(imgs[:, 0, :, :], dim=1)  # 恢复通道维度
        moving_img = imgs[:, 1:, :, :]
        fixed_seg = torch.unsqueeze(segs[:, 0, :, :], dim=1)
        moving_seg = segs[:, 1:, :, :]
        with torch.set_grad_enabled(False):  # 控制梯度计算的开启关闭
            pred_flows, warped_imgs, warped_segs , warped_grid = model(imgs, segs)
            photo_loss, corr_loss, smooth_loss, loss = criterion(pred_flows, warped_imgs, fixed_img)
            # imgs 包含前1通道fixed 后1 moving

        batch_time = time.time() - tic
        tic = time.time()
        avg_photo_loss.update(photo_loss.item())
        avg_smooth_loss.update(smooth_loss.item())
        avg_corr_loss.update(corr_loss.item())
        avg_loss.update(loss.item())
        avg_batch_time.update(batch_time)


        # Metrics calculate and save
        for j in range(fixed_img.size(0)):
            avg_dice.update(dice_average(fixed_seg[j,0,:,:], warped_segs[j, 0, :, :]))
            avg_mse.update(MSE(fixed_img[j,0,:,:], warped_imgs[0][j,0,:,:]))
            avg_psnr.update(PSNR(fixed_img[j,0,:,:], warped_imgs[0][j, 0, :, :]))
            avg_ssim_img.update(structural_similarity(fixed_img[j,0,:,:].detach().cpu().numpy(),warped_imgs[0][j, 0, :, :].detach().cpu().numpy(),data_range=1.0))
            avg_ssim_seg.update(structural_similarity(fixed_seg[j,0,:,:].detach().cpu().numpy(), warped_segs[j, 0, :, :].detach().cpu().numpy(),data_range=1.0))
            avg_hausdorff.update(dist_hausdorff(fixed_seg[j, 0, :, :], warped_segs[j, 0, :, :]))
            avg_mi.update(MI(fixed_img[j,0,:,:], warped_imgs[0][j, 0, :, :]))
            avg_corr.update(CORR(fixed_img[j,0,:,:], warped_imgs[0][j, 0, :, :]))

        mag, mag_colored = flow_mag(pred_flows[0].cpu())
        avg_mag.update(mag)
        writer.add_images("fixed_img", fixed_img, i + 1)
        writer.add_images("moving_img", moving_img, i + 1)
        writer.add_images("warped_img", warped_imgs[0], i + 1)
        writer.add_images("warped_grid", warped_grid, i + 1)
        writer.add_scalar('dice_avg', avg_dice.avg, i+ 1)
        writer.add_scalar('dice_single', avg_dice.data[i], i + 1)
        writer.add_scalar('mse_avg', avg_mse.avg, i + 1)
        writer.add_scalar('mse_single', avg_mse.data[i], i + 1)
        writer.add_scalar('psnr_avg', avg_psnr.avg, i + 1)
        writer.add_scalar('psnr_single', avg_psnr.data[i], i + 1)
        writer.add_scalar('hausdorff_avg', avg_hausdorff.avg, i + 1)
        writer.add_scalar('hausdorff_single', avg_hausdorff.data[i], i + 1)
        writer.add_scalar('ssim_img', avg_ssim_img.avg, i + 1)
        writer.add_scalar('ssim_img_single', avg_ssim_img.data[i], i + 1)
        writer.add_scalar('ssim_seg', avg_ssim_seg.avg, i + 1)
        writer.add_scalar('ssim_seg_single', avg_ssim_seg.data[i], i + 1)
        writer.add_scalar('mi_avg', avg_mi.avg, i + 1)
        writer.add_scalar('mi_single', avg_mi.data[i], i + 1)
        writer.add_scalar('corr_avg', avg_corr.avg, i + 1)
        writer.add_scalar('corr_single', avg_corr.data[i], i + 1)
        writer.add_scalar('mag_avg', avg_mag.avg, i + 1)
        writer.add_images('flow_mag',mag_colored,i+1)

        if i % PRINT_INTERVAL == 0 or i+1 == int(length/args.batch_size):
            if i < int(length/args.batch_size):
                print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'smooth_loss {smooth.val:5.4f} ({smooth.avg:5.4f})\t'
                      'corr_loss {corr.val:5.4f} ({corr.avg:5.4f})\t'
                      'photo_loss {photo.val:5.4f} ({photo.avg:5.4f})'.format(
                    "EVAL", i+1, int(length/args.batch_size), batch_time=avg_batch_time, loss=avg_loss,
                    smooth=avg_smooth_loss, photo=avg_photo_loss, corr=avg_corr_loss))
        i += 1

    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg smooth_loss {smooth.avg:5.4f} \t'
          'Avg corr_loss {corr.avg:5.4f} \t'
          'Avg photo_loss {photo.avg:5.4f} \n'.format(
        batch_time=int(avg_batch_time.sum), loss=avg_loss,
        smooth=avg_smooth_loss, photo=avg_photo_loss, corr=avg_corr_loss))


    return avg_photo_loss.avg, avg_corr_loss.avg, avg_smooth_loss.avg, avg_loss.avg


if __name__ == '__main__':
    # 参数读取
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default="C:/OASIS1/masked", type=str, metavar='DIR_Img',
                        help='path to dataset')  # 修改
    parser.add_argument('--seg_dir', default="C:/OASIS1/seg", type=str, metavar='DIR_Seg',
                        help='path to dataset')  # 修改
    parser.add_argument('--model', default='flownet2', type=str, help='the model to be trained with ('
                                                                     'flownets, flownet2, pwc, raft)')  # 修改
    parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size')
    args = parser.parse_args()

    seed_everything(8)
    # Load Model
    OFEmodel = opticalFlowReg(conv_predictor=args.model)
    OFEmodel.to(device)

    path = os.path.join("Unsupervised", type(OFEmodel.predictor).__name__)
    loss_fnc = OFEloss

    # load self trained model
    checkpoint = torch.load(os.path.join("model_weight", path, 'best_weight.pt'), map_location=device)
    OFEmodel.load_state_dict(checkpoint['model_state_dict'])

    # load flyingchairs pretrained pwc model
    # pt = torch.load(r"C:\Users\13660\PycharmProjects\OFE-Reg\PWC\pwc_net_chairs.pth.tar", map_location=device)
    # pt['conv1a.0.weight'] = pt['conv1a.0.weight'].sum(1, keepdim=True)
    # OFEmodel.load_state_dict(pt, strict=False)

    # load flyingchairs pretrained flownets model
    # pt = torch.load(r"C:\Users\13660\PycharmProjects\OFE-Reg\FlowNetS\flownets_bn_EPE2.459.pth.tar",
    #                 map_location=device)
    # weight = pt['state_dict']['conv1.0.weight']
    # sum1 = weight[:, :3, :, :].sum(dim=1, keepdim=True)
    # sum2 = weight[:, 3:, :, :].sum(dim=1, keepdim=True)
    # new_weight = torch.cat([sum1, sum2], dim=1)
    # pt['state_dict']['conv1.0.weight'] = new_weight
    # OFEmodel.load_state_dict(pt, strict=False)

    # load pretrained flownet2 model
    # pt = torch.load(r"C:\Users\13660\PycharmProjects\OFE-Reg\flownet2\FlowNet2_checkpoint.pth.tar", map_location=device)
    # pt['conv1a.0.weight'] = pt['conv1a.0.weight'].sum(1, keepdim=True)
    # OFEmodel.load_state_dict(pt['state_dict'], strict=False)

    # Data prepare

    # real
    fiximg_dir = "C:/OASIS1/fiximg"
    fixseg_dir = "C:/OASIS1/fixseg"
    movimg_dir = "C:/OASIS1/movimg"
    movseg_dir = "C:/OASIS1/movseg"

    eval_generator, eval_length = real_ds(fiximg_dir=fiximg_dir, fixseg_dir=fixseg_dir,
                                          movimg_dir=movimg_dir, movseg_dir=movseg_dir, batch_size=1)

    # simulation
    # data_dir = args.img_dir
    # seg_dir = args.seg_dir
    # eval_generator, eval_length = eval_random_ds(data_dir=data_dir, seg_dir=seg_dir, batch_size=args.batch_size)
    # train_generator, val_generator, test_generator, train_length, val_length = volume_ds(data_dir=data_dir, batch_size=args.batch_size)

    # generate tensorboard log
    writer = SummaryWriter("./log_fn2_real_ft", flush_secs=30)


    # Evaluate start

    print("=================\n EVAL Start " + " \n=================\n")

    photo_loss, corr_loss, smooth_loss, total_loss = epoch(OFEmodel, eval_generator, loss_fnc)


    # with torch.no_grad():
    #     OFEmodel.eval()
    #     pred_flow = OFEmodel.predictor()

    writer.add_scalars('loss', {"eval": total_loss}, 1)
    writer.add_scalars('photo_loss', {"eval": photo_loss}, 1)
    writer.add_scalars('corr_loss', {"eval": corr_loss}, 1)
    writer.add_scalars('smooth_loss', {"eval": smooth_loss}, 1)

    writer.close()
    print("---------Train complete!---------")
