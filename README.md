# Super-Resolution Neural Operator

This repository contains the official implementation for SRNO introduced in the following paper:

[**Super-Resolution Neural Operator**](https://arxiv.org/abs/2303.02584) (CVPR 2023)

Our code is based on Ubuntu 18.04, pytorch 1.10.2, CUDA 11.3 and python 3.9.

## Train
`python train.py --config configs/train_edsr-sronet.yaml`
if you want to change encoder, please modify the yaml file

```yaml
model:
  name: sronet
  args:
    encoder_spec:
      name: edsr-baseline ## or rdn
      args:
        no_upsampling: true
    width: 256
    blocks: 16
```

## Test
Download a DIV2K pre-trained model.

Model|Download
:-:|:-:
EDSR-baseline-SRNO|[Google Drive](https://drive.google.com/file/d/10eoYPpmR1mXgmWU9eptvfgYEpQehhhIz/view?usp=sharing)
RDN-SRNO|[Google Drive](https://drive.google.com/file/d/12RL7b5ZAz7iKdyuAD7Wfy15ntZNno4RP/view?usp=sharing)

`python test.py --config configs/test_srno.yaml --model edsr-baseline_epoch-1000.pth --mcell True`

## Demo
`python demo.py --input input.png --model save/edsr-baseline_epoch-1000.pth --scale 2 --output output.png`

## Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Wei_2023_CVPR, 
author = {Wei, Min and Zhang, Xuesong}, 
title = {Super-Resolution Neural Operator}, 
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
month = {June}, 
year = {2023}, 
pages = {18247-18256}
}
```

## Acknowledgements
This code is built on [LIIF](https://github.com/yinboc/liif) and [LTE](https://github.com/jaewon-lee-b/lte)
