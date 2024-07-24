# Self-supervised Medical Image Registration using Deep Optical Flow Estimation with Unlabeled Brain MRI Data



## Usage
### Prerequisites
Following libraries are necessary:

- Python = 3.9.18
- PyTorch = 2.0.0
- numpy
- Monai
- Scikit-Image
- ANTsPy
- SciPy
- TorchMetrics

The custom layers should be installed.
Please refer: https://github.com/NVIDIA/flownet2-pytorch


### Train
Train with different models:
```bash
python train.py --img_dir [dir_to_img_data] --model [type_of_model] --epochs [number_of_epochs] --batch_size 24 --lrIni 1e-4
```

### Evaluation
Evaluate with different models:
```bash
python inference.py --img_dir [dir_to_img_data] --model [type_of_model]
```
