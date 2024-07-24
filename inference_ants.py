import argparse
import os
import torch
from models import opticalFlowReg
from utils import averageMeter, dice_average,seed_everything, dist_hausdorff,MSE, PSNR, MI,CORR, flow_mag,grid_generator
from loss import OFEloss
import time
from dataset import eval_random_ds,real_ds
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity
import ants


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 调用gpu计算如果可用，否则cpu
device = torch.device('cuda')
# 打印间隔
PRINT_INTERVAL = 2


def epoch(data):
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
    length = eval_length


    i = 0
    tic = time.time()

    for imgs in data:  # imgs 包含前1通道fixed 后1 moving
        # imgs = imgs.to(device)
        segs = imgs["seg_c"]
        imgs = imgs["image_c"]
        # imgs = imgs.float()
        fixed_img = ants.from_numpy(imgs[0, 0, :, :].numpy())
        moving_img = ants.from_numpy(imgs[0, 1, :, :].numpy())
        fixed_seg = ants.from_numpy(segs[0, 0, :, :].numpy())
        moving_seg = ants.from_numpy(segs[0, 1, :, :].numpy())
        grid = ants.from_numpy(grid_generator().numpy())
        ants_result = ants.registration(fixed_img, moving_img, mask=fixed_seg, moving_mask=moving_seg,
                                        type_of_transform='SyNOnly',reg_iterations=(10, 0, 0))
        ants_warped_img = ants.apply_transforms(fixed=fixed_img, moving=moving_img,
                                                transformlist=ants_result['fwdtransforms'])
        ants_warped_seg = ants.apply_transforms(fixed=fixed_seg, moving=moving_seg,
                                                transformlist=ants_result['fwdtransforms'],
                                                interpolator='nearestNeighbor')
        # ants_warped_grid = ants.apply_transforms(fixed=fixed_seg, moving=grid,
        #                                         transformlist=ants_result['fwdtransforms'],
        #                                         interpolator='nearestNeighbor')





        batch_time = time.time() - tic
        tic = time.time()
        avg_batch_time.update(batch_time)

        # Metrics calculate and save
        avg_mse.update(MSE(torch.from_numpy(fixed_img.numpy()) , torch.from_numpy(ants_warped_img.numpy())))
        avg_psnr.update(PSNR(torch.from_numpy(fixed_img.numpy()) , torch.from_numpy(ants_warped_img.numpy())))
        avg_mi.update(MI(torch.from_numpy(fixed_img.numpy()) , torch.from_numpy(ants_warped_img.numpy())))
        avg_corr.update(CORR(torch.from_numpy(fixed_img.numpy()) , torch.from_numpy(ants_warped_img.numpy())))
        avg_dice.update(dice_average(torch.from_numpy(fixed_seg.numpy()) , torch.from_numpy(ants_warped_seg.numpy())))
        avg_hausdorff.update(dist_hausdorff(torch.from_numpy(fixed_seg.numpy()) , torch.from_numpy(ants_warped_seg.numpy())))
        avg_ssim_img.update(structural_similarity(fixed_img.numpy(),ants_warped_img.numpy(),data_range=1.0))
        avg_ssim_seg.update(structural_similarity(fixed_seg.numpy(), ants_warped_seg.numpy(),data_range=1.0))


        writer.add_images("fixed_img", imgs[:1, :1, :, :], i + 1)
        writer.add_images("moving_img", imgs[:1, 1:, :, :], i + 1)
        writer.add_images("warped_img", torch.from_numpy(ants_warped_img.numpy()).view(1,1,256,256), i + 1)
        # writer.add_images("warped_grid", torch.from_numpy(ants_warped_grid.numpy()).view(1, 1, 256, 256), i + 1)
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


        if i % PRINT_INTERVAL == 0 or i+1 == int(length/args.batch_size):
            if i < int(length/args.batch_size):
                print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'.format(
                    "EVAL", i+1, int(length/args.batch_size), batch_time=avg_batch_time ))
        i += 1

    print('\n===============> Total time {batch_time:d}s\n'.format(
        batch_time=int(avg_batch_time.sum)))

    return 0


if __name__ == '__main__':
    # 参数读取
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default="C:/OASIS1/masked", type=str, metavar='DIR_Img',
                        help='path to dataset')  # 修改
    parser.add_argument('--seg_dir', default="C:/OASIS1/seg", type=str, metavar='DIR_Seg',
                        help='path to dataset')  # 修改
    parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size')
    args = parser.parse_args()

    seed_everything(8)
    # Data prepare

    # real
    # fiximg_dir = "C:/OASIS1/fiximg"
    # fixseg_dir = "C:/OASIS1/fixseg"
    # movimg_dir = "C:/OASIS1/movimg"
    # movseg_dir = "C:/OASIS1/movseg"
    #
    # eval_generator, eval_length = real_ds(fiximg_dir=fiximg_dir, fixseg_dir=fixseg_dir,
    #                                       movimg_dir=movimg_dir, movseg_dir=movseg_dir, batch_size=1)

    data_dir = args.img_dir
    seg_dir = args.seg_dir
    eval_generator, eval_length = eval_random_ds(data_dir=data_dir, seg_dir=seg_dir, batch_size=args.batch_size)

    # generate tensorboard log
    writer = SummaryWriter("./log_ants_64", flush_secs=30)


    # Evaluate start

    print("=================\n EVAL Start " + " \n=================\n")

    epoch(eval_generator)

    writer.close()
    print("---------Train complete!---------")
