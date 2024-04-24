
# EECS 556 Guided Image Colorization

This repository is an attempted recreation of [Guided Colorization Using Mono-Color Image Pairs](https://ieeexplore.ieee.org/document/10017185). 

![res](https://github.com/Tongyuan-Qian/EECS_556_Proj/assets/58116786/dd4ebd0e-776e-476c-978a-4692d40b1263)



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


The Middlebury dataset is only 30 images, and it is contained in ALL-2views. One can also find the original dataset with larger images at https://vision.middlebury.edu/stereo/data/ .
## Training/Evaluation

To run out model, all one has to do is run the colorize.py. Given we only used the Middlebury 2005-2006 dataset, the current code is designed to iterate over all the fules in the ALL-2views directory, so no changes need to be made. Evaluation metrics for each image and overall evaluation metrics are outputted to the command line, but could be saved easily to a csv by adding data storage commands ~lines 129 (each image evaluation) and at the end of file. 

Hyperparamters:
The main hyperparameters for the model (we didn't up testing with noise as we tried to fix the incoherent colorization) are: 

S (Patch Size), default = 16
Rho (Sample Rate, determines Step), default = N / S ** 2
W_h (Horizontal Search Window Size), default = 100 (for simplicity, original paper bases it off maximum difference throughout image)
W_v (Vertical Search Window Size), default = 30
N (Max Number of Dense Scribbling Matches), default = 5
T (Threshold of Matches to determine good colorization), default = 5

Note:  [Guided Colorization Using Mono-Color Image Pairs](https://ieeexplore.ieee.org/document/10017185) shows that N=T=5 leads to optimal colorization. 

These hyperparameters can be found in colorize.py ~lines 47-51

evaluation.py also has some hyperparameters:
sigmaX (For Gaussian Weighted Mean) = 1 [Image-Difference Prediction: From Grayscale to Color](\https://ieeexplore.ieee.org/document/6307862) 's MATLAB implementation is on a deprecated website, so we picked reasonable hyperparamaters, but they are subject to change. 

a and c hyperparameters are provided in the paper above. 

S (Patch Size), default = 15. This is needed to be odd for the Gaussian weighted mean. Given we used the smaller image dataset, any patch size smaller than 15 leads to issues with multiscale CID's downsampling to 1x1 patches. This led to a N=3 multiscale CID. 

Step, default = 10. Step hyperparameter for CID calculation not given, picked 10 for runtime speed up while not sacrificing overall metric. 

PSNR, SSIM, DeltaE_cie76 (Average Euclidean distance between each pixel's LAB), CID outputted for each image and for full dataset in command line by running colorize.py with no input parameters. 






## Results

Our model achieves the following performance on the Middlebury Dataset :


| Model name          | PSNR ^ | SSIM ^ | DeltaE v | CID v |
| ------------------  |--------| -------|--------- |-------|
| Colorization Model  |  15.05 |  .367  |   60.56  | .4947 |
| Original Work       |  37.29 |  .989  |    2.13  | .0265 |





