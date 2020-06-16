# Vgg-16
VGG16 is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. It is considered to be one of the excellent vision model architecture till date. Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.

[link to VGG16 paper!](https://arxiv.org/pdf/1409.1556.pdf)

In this repo i will create VGG16 Architecture in Tensorflow in two differet ways.
* Functional API
* Custom Subclassing

# VGG16 Architecture
![vgg16](https://user-images.githubusercontent.com/50628520/84739567-9ae2d400-afcb-11ea-96ad-48956cd16520.png)


# Configuration
![vgg16 (1)](https://user-images.githubusercontent.com/50628520/84739627-ad5d0d80-afcb-11ea-824a-ee661c4d661c.png)

# My model:
![vgg16_model](https://user-images.githubusercontent.com/50628520/84750005-72160b00-afda-11ea-8232-7a18e6559bc6.png)
