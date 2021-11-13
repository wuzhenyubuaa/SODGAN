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
3. Run **Step2: Sampling** to synthesize annotation-image pairs.
4. Run **Step3: Train SOD model**.


#### 1. Training Mask Generator

```
python train_interpreter.py --exp experiments/<exp_name>.json 



## License

For any code dependency related to StyleGAN, StyleGAN2, and BigGAN, the license is under the [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license by NVIDIA Corporation.  To view a copy of this license, visit [LICENSE](https://github.com/NVlabs/stylegan/blob/master/LICENSE.txt ).
