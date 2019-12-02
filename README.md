## Block-wise Scrambled Image Recognition Using Adaptation Network

This repository contains a Pytorch implementation of Proposed Adaptation Network in our AAAI WS 2020 paper:

Koki Madono, Masayuki Tanaka, Masaki Onishi, Tetsuji Ogawa. Block-wise Scrambled Image Recognition Using Adaptation Network. In AAAI WS, 2020
<!-- [Arxiv] () -->

Adaptation Network is described in Section "Adaptation Network for Block-WiseScrambled Image Recognition"
<!-- <!-- Muhammed Kocabas, Salih Karagoz, Emre Akbas. MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network. In ECCV, 2018. [Arxiv](https://arxiv.org/abs/1807.04067) - -->

### Getting Started
We have tested our method on [cifar10/100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

### Prerequisites
```
python
pytorch
numpy
scikit-image
```

### Installing

1. Clone this repository: 
`git clone https://github.com/MADONOKOUKI/aaai_ws.git`

2. Install [Pytorch](https://pytorch.org/).

3. ```pip install -r src/requirements.txt```

<!-- 4. To download COCO dataset train2017 and val2017 annotations run: `bash data/coco.sh`. (data size: ~240Mb)

## Training

`python main.py`

For more options take a look at `opt.py`

## Results
Results on COCO val2017 Ground Truth data.

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.894
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.971
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.912
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.875
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.918
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.909
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.972
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.928
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.896
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.947
```

## License

## Other Implementations

[Pytorch Version](https://github.com/salihkaragoz/pose-residual-network-pytorch) --> --> -->

<!-- 
## Citation
If you find this code useful for your research, please consider citing our paper:
```
@Inproceedings{kocabas18prn,
  Title          = {Multi{P}ose{N}et: Fast Multi-Person Pose Estimation using Pose Residual Network},
  Author         = {Kocabas, Muhammed and Karagoz, Salih and Akbas, Emre},
  Booktitle      = {European Conference on Computer Vision (ECCV)},
  Year           = {2018}
}
``` -->
