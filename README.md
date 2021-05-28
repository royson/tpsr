# Journey Towards Tiny Perceptual Super-Resolution

Test code for our ECCV2020 paper: <https://arxiv.org/abs/2007.04356>

Our x4 upscaling pre-trained models, namely TPSR_NOGAN, TPSR_ESRGAN, and TPSR_D2, are in the folder 'pretrained'.

The `evaluate.py` script runs the evaluation pipeline on the images in 'input_images' ('Set5' by default) and evaluates the PSNR against the ground truth in 'ground_truth' folder.

Upsampled images are saved in 'output_images'. 

#### Examples:

1) Evaluating PSNR using TPSR_D2 on Set5:
`python evaluate.py --model pretrained/TPSR_D2X4.pt`

2) Getting output images using your own input images in the 'input_images' folder using TPSR_D2 (No evaluation)
`python evaluate.py --model pretrained/TPSR_D2X4.pt --ground_truth ''`

#### Citation:

Please consider citing our paper if you find it helpful:
```
@article{Lee2020JourneyTT,
  title={Journey Towards Tiny Perceptual Super-Resolution},
  author={Royson Lee and L. Dudziak and M. Abdelfattah and Stylianos I. Venieris and H. Kim and Hongkai Wen and N. Lane},
  journal={ECCV},
  year={2020}
}
```

