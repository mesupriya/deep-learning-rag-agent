# Artificial Neural Networks (ANN)

## The Perceptron
The artificial neuron, or perceptron, is the fundamental building block of an Artificial Neural Network. It computes a weighted sum of its inputs, adds a bias term, and passes the result through an activation function. This mathematical model is inspired by biological neurons in the human brain. While a single perceptron can only solve linearly separable problems, combining them into networks allows for learning highly complex patterns.

## Feedforward Networks
A feedforward neural network consists of an input layer, one or more hidden layers, and an output layer. Information flows in only one direction, from input to output, without any loops or cycles. Each neuron in a layer is typically connected to all neurons in the subsequent layer, creating a fully connected architecture. These networks are universal approximators, meaning they can represent any continuous function given sufficient width and depth.

## Backpropagation
Backpropagation is the primary algorithm used to train artificial neural networks. It works by computing the gradient of the loss function with respect to every weight in the network using the chain rule of calculus. The algorithm operates in two phases: a forward pass that computes the network's predictions and loss, and a backward pass that propagates the error gradients from the output layer back to the input layers. These gradients are then used by an optimizer like Stochastic Gradient Descent (SGD) or Adam to update the weights and minimize the error.
