# ML_SRCNN-Super-Resolution-Convolutional-Neural-Network
Implementation of SRCNN architecture using Deep Learning and PyTorch

**Members** : <a href="https://github.com/nikitaoltyan">Nikita Oltyan</a>, <a href="https://github.com/arseniyzu">Arseniy Zubenya</a>

## Description

This project aims to improve image quality (image debluring) with the use of SRCNN model. The below image briefly explains the output we want:

(HERE WILL BE IMAGE)


## Table of Content

* [Dataset](#dataset)

* [Model](#model)

* [PSNR Function](#psnr_function)

* [Results](#results)

* [Dependency](#dependency)

* [References](#references)


## Dataset <a name="dataset"></a>

The dataset contains:
** 13056 train images (256x128) **
** 3200 val images (256x128) **

You can find a whole lot of image dataset mainly used for super-resolution experimentation in this public <a href="https://drive.google.com/file/d/1QI3MvHTxFzwZfF1xdgJv0EJqB91yAzMG/view?usp=sharing">Google Drive folder</a>.

**All examples of training and validation data are groped into one picture with a good quality one on the left side and a poor quality on the right.**

(HERE WILL BE AN IMAGE OF TRAIN IMAGE)

Below the line you can find a story of an idea of that dataset and how we get that data.

(HERE WILL BE A LINE)

After brief examination of our task we decided that there wasn’t any easy way of implementing Super Resolution for any image of any object we want.

We decided to concentrate our efforts on improving quality of skyline pictures. This is a common problem for a millions of traveling people – they often zoom horizon, trying to take their best pictures of sunsets/city skylines/historical sites.

(HERE WILL BE AN EXAMPLE OF CODE THAT DECREASING QUALITY)

We’ve started with dataset of skyline pictures taken by Nikita from his flat using digital slr but they were too big for any existing models: 3024x4032 pixels.

So we resized them into 1024x1024, bet it still was too big and expensive for resources:

(HERE WILL BE AN IMAGE OF ORIGINAL DATASET)

So we tried to crop them into batches of images with 256x256 size but is was hopeless. 

## Model <a name="model"></a>

## PSNR Function <a name="psnr_function"></a>

## Results <a name="results"></a>

## Dependency <a name="dependency"></a>

## References <a name="references"></a>
