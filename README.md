
# My Paper Title

This repository is an attempted recreation of [Guided Colorization Using Mono-Color Image Pairs](https://ieeexplore.ieee.org/document/10017185). 

![res](https://github.com/Tongyuan-Qian/EECS_556_Proj/assets/58116786/dd4ebd0e-776e-476c-978a-4692d40b1263)



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

The Middlebury dataset is only 30 images, and it is contained in ALL-2views. One can also find the original dataset with larger images at https://vision.middlebury.edu/stereo/data/ .
## Training/Evaluation

To run out model, all one has to do is run the colorize.py. Given we only used the Middlebury 2005-2006 dataset, the current code is designed to iterate over all the fules in the ALL-2views directory, so no changes need to be made. Evaluation metrics for each image and overall evaluation metrics are outputted to the command line, but could be saved easily to a csv by adding data storage commands ~lines 129 (each image evaluation) and at the end of file. 

Hyper

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
