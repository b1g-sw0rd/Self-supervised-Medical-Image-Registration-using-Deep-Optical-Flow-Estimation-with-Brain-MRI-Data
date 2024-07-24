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

### Train & Evaluation
Train the detector with pre-trained model:
```bash
torchpack dist-run -np [number of gpus] python tools/train.py [path_to_config] --load_from [path_to_pretrained]
```

Evaluate the detector with specific type (e.g. bbox) and save results:
```bash
torchpack dist-run -np [number of gpus] python tools/test.py [path_to_config] [path_to_checkpoint.pth] --eval [evaluation type] --out [path_to_result.pkl]
```

Convert the format of detection result to adapt with tracker:
```bash
cd script/
python format_for_tracker.py
```
Note that path of detection result and destination need to be specified in the file.

Run the CenterPoint tracker:
```bash
python track/centerpoint/track_test.py --checkpoint [path_to_detection_result.json]  --work_dir [path_to_result] --bbox-score [confidence e.g. 0.01]
```

Run the Poly-MOT tracker:
```bash
python test.py --detection_path [path_to_detection_result.json] --eval_path [path_to_result] 
```


### Visualization
Visualize detection result:
```bash
torchpack dist-run -np [number of gpus] python tools/visualize.py [path_to_config] --checkpoint [path_to_checkpoint.pth] --out-dir [path_to_vis_result] --mode [pred or gt] --bbox-score [confidence e.g. 0.01]
```

We need to convert data format using (Note that path of detection result and destination need to be specified in the file):
```bash
cd script/
python format_for_tracking_visual.py
```

Then, we can visualize tracking results:
```bash
python tools/track/visualize_tracking.py [path_to_config] --checkpoint [path_to_tracking_result.pkl] --out-dir [path_to_vis_result] --mode [pred or gt] --bbox-score [confidence e.g. 0.01]
```

