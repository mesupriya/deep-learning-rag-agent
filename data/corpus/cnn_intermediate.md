# Convolutional Neural Networks (CNN)

## Convolutional Layers
Convolutional Neural Networks utilize convolutional layers to automatically extract spatial hierarchies of features from grid-like data such as images. Instead of using fully connected layers, a CNN applies a set of learnable filters (or kernels) across the input volume. As the filter slides horizontally and vertically, it computes dot products to produce a 2D activation map that highlights specific features like edges, textures, and eventually complex objects in deeper layers.

## Pooling Layers
Pooling, or subsampling, layers are interspersed between convolutional layers to progressively reduce the spatial dimensions of the network's internal representations. Max pooling is the most common technique, which involves sliding a window over the input and taking the maximum value within that region. This operation significantly reduces the number of parameters and computation required, while also providing translation invariance, meaning the network can recognize features regardless of their exact position in the input.

## Receptive Field
The receptive field of a neuron in a CNN refers to the specific region of the original input image that influences its activation. In early layers, neurons have small receptive fields and capture local low-level features such as corners and straight lines. As the network gets deeper, pooling operations and stacked convolutions cause the receptive field of neurons in higher layers to grow exponentially, allowing them to comprehend global patterns and detect the overall semantic content of the image.
