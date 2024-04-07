__**DAY 11**__

A. Deep Computer Vision
Image Classification & Object Detection/Recognition using deep computer vision with CONVOLUTIONAL NEURAL NETWORK.
Image data as features & Label for those images as label or output.
Concepts:
1. Image Data
2. Convolutional Layer
3. Pooling Layer
4. CNN Architectures

Image Data
3 Dimensions :- Image Height, Image Width, Color Channels

Color Channel - In the context of Deep Computer Vision, a color channel refers to the different components that make up a color image. For instance, in a standard RGB (Red, Green, Blue) image, there are three color channels, each representing the intensity of red, green, or blue in the image. These channels are stacked together to form the complete color image. So, in the context of the provided information, the color channels would be the third dimension of the image data, with the first two dimensions being the height and width of the image.In some cases, images may also be represented as grayscale, which only has a single channel representing the intensity of light.
For each pixel, we have three numeric values in the range 0-255 that define its colour.

Convolutional Neural Network (Convnet)
- It is a type of neural network used to process image data and extract useful information from it.
- The main idea behind convolutional networks is to take an input image (or set of images) and apply a series of filters to this image.
Each convnet is made up of one or many convolutional layers. These are different than dense layers.
Fundamental difference is that dense layers detect patterns globally while convolutional layers detects patterns locally. Wehn we have a densely connected layer, each node in that layer sees all the data from the previous layer. THis means that this layer is looking at ALL of the information and is capable of only analysing the data in a global capacity. The convolutional layer will not be densely connected, this means it can detect local patterns using part of the input data to that layer.

For eg: Dense Layer looks for eyes in only a specific location. Convnet looks for eyes in the whole picture of the dog.

Feature Maps:
3D tensor model with two spacial axes(height, width), and depth representing color channels. These layers take feature maps as their input and return a new feature map that repreents the prescence of specific filters from the previous feature map. These are called response maps.

Layer Parameters: 2 Key Parameters
1. Filters - A m*n pattern of pixels that we are looking for in an image. No. of filters in a convnet represents how many patterns each layer is looking for and what the depth of our response map will be. 
2. Stride: The number of pixels by which the filter moves each time it is applied to the input data. A larger stride results in a smaller output feature map, while a smaller stride results in a larger output feature map.