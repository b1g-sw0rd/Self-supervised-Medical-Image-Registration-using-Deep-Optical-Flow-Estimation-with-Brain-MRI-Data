import argparse
from torch.utils import data
import os
import torch
from models import opticalFlowReg
from utils import averageMeter
from loss import OFEloss
import time
from dataset import brain_dataset, volume2slices_ds
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 调用gpu计算如果可用，否则cpu
device = torch.device('cuda')
# 打印间隔
PRINT_INTERVAL = 10


def epoch(model, data, criterion, optimizer=None, mode="TRAIN"):
    model.eval() if optimizer is None else model.train()
    avg_loss = averageMeter()
    avg_batch_time = averageMeter()
    avg_smooth_loss = averageMeter()
    avg_photo_loss = averageMeter()
    avg_corr_loss = averageMeter()
    if mode == "TRAIN":
        length = train_length
    else:
        length = val_length

    i = 0
    tic = time.time()

    for imgs in data:  # imgs 包含前1通道fixed 后1 moving
        # imgs = imgs.to(device)
        imgs = imgs["image_c"].as_tensor().to(device)
        imgs = imgs.float()
        fixed_img = torch.unsqueeze(imgs[:, 0, :, :], dim=1)  # 恢复通道维度
        with torch.set_grad_enabled(optimizer is not None):  # 控制梯度计算的开启关闭
            pred_flows, wraped_imgs = model(imgs)
            photo_loss, corr_loss, smooth_loss, loss = criterion(pred_flows, wraped_imgs, fixed_img)
            # imgs 包含前1通道fixed 后1 moving
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()  # lr decay

        batch_time = time.time() - tic
        tic = time.time()
        avg_photo_loss.update(photo_loss.item())
        avg_smooth_loss.update(smooth_loss.item())
        avg_corr_loss.update(corr_loss.item())
        avg_loss.update(loss.item())
        avg_batch_time.update(batch_time)

        if i % PRINT_INTERVAL == 0 or i+1 == int(length/args.batch_size):
            if i < int(length/args.batch_size):
                print('[{0:s} Batch {1:03d}/{2:03d}]\t'
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'smooth_loss {smooth.val:5.4f} ({smooth.avg:5.4f})\t'
                      'corr_loss {corr.val:5.4f} ({corr.avg:5.4f})\t'
                      'photo_loss {photo.val:5.4f} ({photo.avg:5.4f})'.format(
                    mode, i+1, int(length/args.batch_size), batch_time=avg_batch_time, loss=avg_loss,
                    smooth=avg_smooth_loss, photo=avg_photo_loss, corr=avg_corr_loss))
        i += 1

    print('\n===============> Total time {batch_time:d}s\t'
          'Avg loss {loss.avg:.4f}\t'
          'Avg smooth_loss {smooth.avg:5.4f} \t'
          'Avg corr_loss {corr.avg:5.4f} \t'
          'Avg photo_loss {photo.avg:5.4f} \n'.format(
        batch_time=int(avg_batch_time.sum), loss=avg_loss,
        smooth=avg_smooth_loss, photo=avg_photo_loss, corr=avg_corr_loss))

    if mode == "TEST":
        writer.add_images("original_img", fixed_img, e + 1)
        writer.add_images("warped_img", wraped_imgs[0], e+1)

    return avg_photo_loss.avg, avg_corr_loss.avg, avg_smooth_loss.avg, avg_loss.avg


if __name__ == '__main__':
    # 参数读取
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default="C:/OASIS1/RAW", type=str, metavar='DIR',
                        help='path to dataset')  # 修改
    parser.add_argument('--model', default='pwc', type=str, help='the supervised model to be trained with ('
                                                                     'flowNetS, flownet2, pwc, raft)')  # 修改
    parser.add_argument('--epochs', default=200, type=int, metavar='E', help='number of epochs')
    parser.add_argument('--batch_size', default=24, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lrIni', default=1e-4, type=float, metavar='LRI', help='learning rate')
    parser.add_argument('--lrMin', default=1e-4, type=float, metavar='LRM', help='learning rate')
    parser.add_argument('--cp', default=True, type=bool, metavar='CP', help='whether to use checkpoint state')

    args = parser.parse_args()
    #  模型加载
    OFEmodel = opticalFlowReg(conv_predictor=args.model)
    OFEmodel.to(device)

    path = os.path.join("Unsupervised", type(OFEmodel.predictor).__name__)
    loss_fnc = OFEloss
    optim = torch.optim.Adam(OFEmodel.parameters(), args.lrIni, betas=(0.9, 0.999), eps=args.lrMin)
    scheduler = StepLR(optim,
                       step_size=20,  # Period of learning rate decay
                       gamma=0.8)  # Multiplicative factor of learning rate decay


    # 准备数据
    # data_dir = r"D:\OASIS"
    # train_generator = data.DataLoader(brain_dataset("D:/test/imgA", "D:/test/imgB"), batch_size=args.batch_size,
    #                                   shuffle=False)
    # val_generator = data.DataLoader(brain_dataset("D:/test/imgA", "D:/test/imgB"), batch_size=args.batch_size,
    #                                   shuffle=True)
    # test_generator = data.DataLoader(brain_dataset("D:/test/imgA", "D:/test/imgB"), batch_size=args.batch_size,
    #                                 shuffle=True)
    # train_length = len(train_generator)
    data_dir = args.root
    train_generator, val_generator, test_generator, train_length, val_length = volume2slices_ds(data_dir=data_dir, batch_size=args.batch_size)

    epochs = args.epochs

    os.makedirs(os.path.join("Checkpoints", path), exist_ok=True)
    os.makedirs(os.path.join("model_weight", path), exist_ok=True)
    writer = SummaryWriter("./log", flush_secs=30)
    starting_epoch = 0
    best_loss = 100000

    if os.path.exists(os.path.join("Checkpoints", path, 'training_state.pt')) and args.cp:
        print("----------loading checkpoints!------------")
        checkpoint = torch.load(os.path.join("Checkpoints", path, 'training_state.pt'), map_location=device)
        OFEmodel.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']


    for e in range(starting_epoch, epochs):
        print("=================\n EPOCH " + str(e + 1) + "/" + str(epochs) + " \n=================\n")
        print("learning rate : ", optim.param_groups[0]["lr"])
        writer.add_scalar('lr', optim.param_groups[0]["lr"], e + 1)
        # train start
        photo_loss, corr_loss, smooth_loss, total_loss = epoch(OFEmodel, train_generator, loss_fnc, optim, mode="TRAIN")

        torch.save({
            'epoch': e,
            'model_state_dict': OFEmodel.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optim.state_dict(),
        }, os.path.join("Checkpoints", path, 'training_state.pt'))

        # validation start
        photo_loss_val, corr_loss_val, smooth_loss_val, total_loss_val = epoch(OFEmodel, val_generator, loss_fnc, mode="VAL")

        if total_loss_val < best_loss:
            print("---------saving new weights!----------")
            best_loss = total_loss_val
            torch.save({
                'model_state_dict': OFEmodel.state_dict(),
                'loss_val': total_loss_val, 'photo_loss_val': photo_loss_val, 'corr_loss_val': corr_loss_val,
                'smooth_loss_val': smooth_loss_val,
                'loss': total_loss, 'photo_loss': photo_loss, 'corr_loss': corr_loss, 'smooth_loss': smooth_loss,
            }, os.path.join("model_weight", path, 'best_weight.pt'))

        photo_loss_test, corr_loss_test, smooth_loss_test, total_loss_test = epoch(OFEmodel, test_generator, loss_fnc, mode="TEST")

        # with torch.no_grad():
        #     OFEmodel.eval()
        #     pred_flow = OFEmodel.predictor()

        writer.add_scalars('loss', {"train": total_loss, "val": total_loss_val, "test": total_loss_test}, e+1)
        writer.add_scalars('photo_loss', {"train": photo_loss, "val": photo_loss_val, "test": photo_loss_test}, e+1)
        writer.add_scalars('corr_loss', {"train": corr_loss, "val": corr_loss_val, "test": corr_loss_test}, e+1)
        writer.add_scalars('smooth_loss', {"train": smooth_loss, "val": smooth_loss_val, "test": smooth_loss_test}, e+1)

    writer.close()
    print("---------Train complete!---------")
