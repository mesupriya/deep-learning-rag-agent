# Recurrent Neural Networks (RNN)

## Sequential Processing
Recurrent Neural Networks are specifically designed to process sequential data, such as time series or natural language text. Unlike feedforward networks, RNNs maintain an internal hidden state that is updated at each time step. By feeding the hidden state from the previous step alongside the current input, the network incorporates information about the past data. This recurrent loop allows the network to model temporal dependencies and variable-length sequences effectively.

## Backpropagation Through Time
Training an RNN requires a specialized variant of backpropagation known as Backpropagation Through Time (BPTT). Because the hidden state is carried forward, the entire sequence must be computationally unrolled into what looks like a deep feedforward network where each layer corresponds to a time step. The gradients are computed starting from the final time step and propagated backward through both the layers and time. This unrolling process poses significant computational and memory challenges for very long sequences.

## Vanishing Gradient Problem
The most notable limitation of standard RNNs is the vanishing gradient problem, which arises when training on long sequences using BPTT. Because gradients are repeatedly multiplied by the same weight matrices during the backward pass, they tend to exponentially shrink toward zero if the weights are small. When gradients vanish, the network fails to update the weights for earlier time steps, making it incapable of learning long-term dependencies.
