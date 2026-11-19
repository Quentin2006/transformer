/*
 * we can use linear algebra to compute the next layers activation
 *
 * quick recap, neural networks are made up of neurons and connections
 *
 * neurons are made up of activations and bias
 *
 * connections connect neurons in adjacent layers to each other with a weight
 *
 * we compute the current layer's activation by multiplying the previous layer's
 * activation by the connection's weight and adding the current layer's bias,
 * then passing it through an activation function (ReLU, sig, ...).
 *
 *
 *
 *
 * using linear algebra, we can compute the next layers activation via the
 * following formula:
 *
 * activation(W_l*A_l+B_l+1) = A_l+1
 *
 * where:
 *    W_l is the weight matrix connecting layer l to layer l+1
 *    A_l are the activations for all neurons in layer l
 *    B_l+1 are the biases for all neurons in layer l+1
 *    activation is our activation function
 *
 *
 * W_l is a kxn matrix, where
 * row i encodes the connections from all neurons in layer l to neuron i in
 * layer l+1.
 *
 * W_l = w_0_0   w_0_1   ... w_0_n-1
 *       w_1_0   w_1_1   ... w_1_n-1
 *                       ...
 *       w_k-1_0 w_k-1_1 ... w_k-1_n-1
 *
 *
 *
 *  and layers look like
 *
 * layer 0      layer 1
 *
 * n_0_0        n_1_0
 * n_0_1        n_1_1
 * n_0_2        n_1_2
 *        ...
 * n_0_n-1      n_1_k-1
 *
 * and that's all a neural net is
 */

#include <nn.hpp>

nn::Scalar nn::NeuralNetwork::ReLU(nn::Scalar a) {
  if (a > 0.)
    return a;
  return 0.;
}

nn::NeuralNetwork::NeuralNetwork(const nn::Config &config) {
  this->config = config;

  switch (config.activation) {
  case ACTIVATION::RELU:
    activationFunction = &NeuralNetwork::ReLU;
    break;
  }

  // alocate layers
  layers.resize(config.hiddenLayerCount +
                2); // +2 due to input and output layers

  for (std::size_t i = 1; i < layers.size(); ++i) {
    // # of neruons in next layers
    const int out =
        (i == layers.size() - 1) ? config.outputDim : config.hiddenDim;

    // # of neurons in prev layer
    const int in = (i == 1) ? config.inputDim : layers[i - 1].weight.rows();

    Layer &layer = layers[i];
    layer.weight.resize(out, in);
    layer.weight.setRandom();

    layer.bias = Vector::Zero(out);

    layer.activations.resize(out);
    layer.activations.setZero();
  }
}
