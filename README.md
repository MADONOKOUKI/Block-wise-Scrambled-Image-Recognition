## Block-wise Scrambled Image Recognition Using Adaptation Network

This repository contains a Pytorch implementation of Proposed Adaptation Network in our AAAI WS 2020 paper:

Koki Madono, Masayuki Tanaka, Masaki Onishi, and Tetsuji Ogawa. Block-wise Scrambled Image Recognition Using Adaptation Network. In AAAI WS, 2020
<!-- [Arxiv] () -->

Adaptation Network is described in Section "Adaptation Network for Block-WiseScrambled Image Recognition"
<!-- <!-- Muhammed Kocabas, Salih Karagoz, Emre Akbas. MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network. In ECCV, 2018. [Arxiv](https://arxiv.org/abs/1807.04067) - -->


![](src/Scrambled_Image_Classification.png)
This network can be used for cloud based machine learning in visual information hiding setting.

![](src/block_scrambling.png)
Scrambling method is described in this figure.
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

<!--  -->
## License

## Citation
If you find this code useful for your research, please consider citing our paper:
```
@Inproceedings{madono2020,
  Title           = {Block-wise Scrambled Image Recognition Using Adaptation Network},
  Author         = {Koki Madono, Masayuki Tanaka, Masaki Onishi, and Tetsuji Ogawa},
  Booktitle      = {AAAI WS},
  Year           = {2020}
}
```

### Reference codes

- [mastnk/ICCE-TW2018](https://github.com/mastnk/ICCE-TW2018)
- [owruby/shake-drop_pytorch](https://github.com/owruby/shake-drop_pytorch)
- [utkuozbulak/pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/inverted_representation.py)

## License

MIT License. Please see the LICENSE file for details.