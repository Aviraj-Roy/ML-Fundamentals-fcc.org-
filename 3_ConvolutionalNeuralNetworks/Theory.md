https://colab.research.google.com/drive/1ZZXnCjFEOkp_KdNcNabd14yok0BAIuwS#forceEdit=true&sandboxMode=true

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


__**DAY 12**__

Concept of Padding
Padding is a concept in deep learning, specifically in the context of convolutional neural networks (CNNs), that refers to the addition of extra pixels around the edges of an image before it is passed through a convolutional layer. The purpose of padding is to preserve the spatial dimensions of the input data as it passes through the network, which can help to prevent the loss of important information and improve the performance of the model.
There are two common types of padding used in CNNs:
1. Valid padding: This is the default padding mode in most deep learning frameworks, and it does not add any extra pixels to the input data. This means that the spatial dimensions of the input data will be reduced as it passes through the convolutional layer.
2. Same padding: This padding mode adds extra pixels around the edges of the input data so that the spatial dimensions of the output data are the same as the input data. This can help to preserve the spatial information in the input data and improve the performance of the model.
Padding is an important concept to understand when working with CNNs, as it can have a significant impact on the performance of the model. By adding padding to the input data, we can help to preserve the spatial dimensions of the data and improve the model's ability to extract useful features from the data.

Concept of Pooling
Pooling is a type of downsampling technique used in convolutional neural networks (CNNs) to reduce the spatial dimensions of the input data while retaining the most important information. This helps to reduce the computational complexity of the model and prevent overfitting. 
There are three main types of pooling used in CNNs:
1. Max pooling: This type of pooling takes the maximum value from a small neighborhood of the input data and uses it as the output. This helps to preserve the most important features of the input data and reduce the spatial dimensions.
2. Average pooling: This type of pooling takes the average value from a small neighborhood of the input data and uses it as the output. This helps to preserve the overall distribution of the input data and reduce the spatial dimensions.
3. Global pooling: This type of pooling is applied to the entire feature map, reducing it to a single value. This is often used in the final layers of a CNN to reduce the spatial dimensions to a single value, which can then be used as input to a fully connected layer.
#Min Pooling also present, not confirmed.
Pooling is an important concept to understand when working with CNNs, as it can help to reduce the computational complexity of the model and prevent overfitting. By downsampling the input data, we can reduce the spatial dimensions while retaining the most important information, which can help to improve the performance of the model.

Q. Three main properties of each convolutional layer
Ans:- Input Size, No. of filters, Sample size of the filters

Working with Small Datasets
- Data Augmentation
To avoid overfitting and create a larger dataset from a smaller one, technique used is called Data Augmentation. Perform random transformations on our images  so that model can generalize better. These transformations can be things like compressions, rotations, stretched and even color changes.


__**DAY 13**__

Pretrained Models
Using a pretrained CNN as a part of our own custom network to imporve the accuracy of the model, giving a good convolutional base.

Fine Tuning
In this technique, we tweak the final layers in the convolutional base to work better for our specific problem. This involves only adjusting the final few layers, and not touching/retraining the earlier layers. 


__**DAY 14**__
Freezing - refers to disabling the training property of a layer. We wont make any changes to the weights of any layers that are frozen during training.