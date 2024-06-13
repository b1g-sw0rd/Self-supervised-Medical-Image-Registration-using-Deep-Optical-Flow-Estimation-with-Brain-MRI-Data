import argparse
from utils import *
from dataset import *
from models import flowNetS, opticalFlowReg
import matplotlib.pyplot as plt
import torch.nn.functional as F
import albumentations as albu
from albumentations.pytorch import ToTensorV2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="D:/test/imgA", type=str, metavar='DIR', help='path to get images')
    parser.add_argument("--unsup", default=True, help="use weights obtained of unsupervised training", action="store_true")
    parser.add_argument('--model', default='flownets', type=str, help='the model to be used either with or without '
                                                                     'supervision (flownet, lightflownet, pwc_net)')
    args = parser.parse_args()
    file_names = sorted(os.listdir(args.path))

    if args.unsup:
        mymodel = opticalFlowReg(conv_predictor=args.model)
        path = os.path.join("Unsupervised", type(mymodel.predictor).__name__)
    else:
        if "light" in args.model:
            mymodel = flowNetS()
        elif "pwc" in args.model:
            mymodel = flowNetS()
        else:
            mymodel = flowNetS()

        path = type(mymodel).__name__

    os.makedirs(os.path.join("result", path), exist_ok=True)
    mymodel.load_state_dict(torch.load(os.path.join("model_weight", path, 'best_weight.pt'), map_location=device)['model_state_dict'])
    if args.unsup:
        mymodel = mymodel.predictor
    mymodel.eval()

    #  Albumentation 图像增强包
    frames_transforms = albu.Compose([
        # albu.Normalize((0., 0., 0.), (1., 1., 1.)),
        albu.ToFloat(),
        ToTensorV2()
    ])

    for i in range(0, len(file_names) - 1, 2):
        frame1 = cv2.imread(os.path.join(args.path, file_names[i]), 0)
        h, w = frame1.shape[:2]
        frame2 = cv2.imread(os.path.join(args.path, file_names[i + 1]), 0)
        # Albumentation 图像增强
        frame1 = frames_transforms(image=frame1)['image']
        frame2 = frames_transforms(image=frame2)['image']
        frames = torch.cat((frame1, frame2), dim=0)
        frames = torch.unsqueeze(frames, dim=0)

        with torch.no_grad():
            flow = mymodel(frames)[0]

        pred_flo = F.interpolate(flow, (h, w), mode='bilinear', align_corners=False)[0]
        pred_flo = computeImg(pred_flo.cpu().numpy(), verbose=True, savePath=os.path.join("result", path,
                                                                                         'predicted_flow' + str(
                                                                                             i//2 + 1) + '.png'))
