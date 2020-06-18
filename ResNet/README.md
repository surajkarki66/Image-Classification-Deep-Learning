# ResNet(Residual Network)  - Kaiming He et al
A residual neural network (ResNet) is an artificial neural network (ANN) of a kind that builds on constructs known from pyramidal cells in the cerebral cortex. Residual neural networks do this by utilizing skip connections, or shortcuts to jump over some layers. Typical ResNet models are implemented with double- or triple- layer skips that contain nonlinearities (ReLU) and batch normalization in between. An additional weight matrix may be used to learn the skip weights; these models are known as HighwayNets.Models with several parallel skips are referred to as DenseNets. In the context of residual neural networks, a non-residual network may be described as a plain network.

It is created by Kaiming He et al

[link to ResNet paper!](https://arxiv.org/abs/1512.03385)

In this repo i will create ResNet-50 Architecture in Tensorflow in two differet ways.
* Functional API
* Custom Subclassing

# ResNet Architecture
![Resnet](https://user-images.githubusercontent.com/50628520/85012661-133fc580-b183-11ea-9783-d82ad4caf783.png)

# Residual Block
<img width="647" alt="Resnet2" src="https://user-images.githubusercontent.com/50628520/85012725-28b4ef80-b183-11ea-94aa-2a1ec79843ff.png">
