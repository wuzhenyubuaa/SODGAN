# SODGAN

This is the official code and data release for:

#### Semi-Supervised Salient Object Detection via Synthetic Data

## Requirements

- Python 3.8  is supported.
- Pytorch 1.8.1.
- This code is tested with CUDA 10.2 toolkit and CuDNN 7.5.

## Training 

To reproduce paper **Semi-Supervised Salient Object Detection via Synthetic Data**: 

```
cd SODGAN
```

1. Run **Step1: training mask generator**.  
3. Run **Step2: synthesizing annotation-image pairs**.
4. Run **Step3: Train SOD model**.


#### 1. Training Mask Generator

we take training stlyegan as an example:

a. Download pretrained model from StyleGAN [https://github.com/NVlabs/stylegan]. Put pretrained model in  'your/path/' and you have revised the path of 'stylegan_checkpoint' of ./experiments/cat_sod.json 

b. Download Dataset from [https://pan.baidu.com/s/1e7SRXVTqTxR3CQJEtq_HFg] (fetch code:2nab ). Unzip stylegan datasets into './data/'

c.
```
python train_stylegan_G_mask.py --exp experiments/stylegan/cat_sod.json  --test False
```


## License

For any code dependency related to StyleGAN, StyleGAN2, and BigGAN, the license is under the [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license by NVIDIA Corporation.  To view a copy of this license, visit [LICENSE](https://github.com/NVlabs/stylegan/blob/master/LICENSE.txt ).
