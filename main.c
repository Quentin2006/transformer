#include <stdlib.h>

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

typedef struct {
  float activation, // same range as your activation fn
      bias;         // [-inf, inf]

} Neuron;

typedef struct {
  int size;
  Neuron *neurons;
} Layer;

typedef struct {
  float weight; // [-inf, inf]
} Connection;

typedef struct {
  Layer *layers;
  Connection ***connections;
  float (*activation)(float);

  int inputDim, hiddenDim, outputDim, numLayers;
  int hiddenLayerCount;
} NeuralNet;

/**
 * @brief will create a neural network inited with random weights [-1,1] and
 * zero biases
 *
 * @param inputDim neurons in layer 0
 * @param hiddenDim neurons in layer [1, n-2]
 * @param outputDim neurons in layer n-1
 * @param hiddenLayerCount # of hidden layers
 * @param activation activation function applied to each layer
 * @return neural net inited with random weights and zero biases
 */
NeuralNet *initNerualNet(int inputDim, int hiddenDim, int outputDim,
                         int hiddenLayerCount, float (*activation)(float)) {
  NeuralNet *res = malloc(sizeof(NeuralNet));

  *res = (NeuralNet){
      .layers = malloc((hiddenLayerCount + 2) *
                       sizeof(Layer)), // +2 for input and output layer
      .connections =
          malloc((hiddenLayerCount + 1) *
                 sizeof(Connection **)), // currently just init it with
                                         // the # of connection layers
                                         // it is only +1 as the last
                                         // layer has no connections
      .activation = activation,
      .inputDim = inputDim,
      .hiddenDim = hiddenDim,
      .outputDim = outputDim,
      .numLayers = hiddenLayerCount + 2,
      .hiddenLayerCount = hiddenLayerCount};

  // now lets allocate each neuron
  res->layers[0].size = inputDim;
  res->layers[res->numLayers - 1].size = outputDim;

  for (int i = 0; i < res->numLayers; ++i) {
    if (i != 0 && i != res->numLayers - 1)
      res->layers[i].size = hiddenDim;

    // doing research and its common to set bias to 0 as its better for training
    res->layers[i].neurons = calloc(res->layers[i].size, sizeof(Neuron));
  }

  // now all the layers neurons are allocated, we can allocate the connections
  // connection[l][k][n] == neuron k in layer l's connection to neuron n in
  // layer l+1

  // for each layer
  for (int l = 0; l < res->numLayers - 1;
       ++l) { // -1 to ignore the last layer as there are no outgoing
              // connections
    // we must allocate one entry per neuron in layer l
    res->connections[l] = malloc(res->layers[l].size * sizeof(Connection *));

    // for each neuron we just allocated, we must allocate one connection per
    // neuron in the next layer
    for (int k = 0; k < res->layers[l].size; ++k) {
      res->connections[l][k] =
          malloc(res->layers[l + 1].size * sizeof(Connection));

      // init the weights w/ value [-1,1]
      for (int n = 0; n < res->layers[l + 1].size; ++n) {
        res->connections[l][k][n].weight = (((float)rand() / RAND_MAX) * 2) - 1;
      }
    }
  }

  return res;
}

void freeNeuralNet(NeuralNet *nn) {

  // first free all connections as it depends on layers
  for (int l = 0; l < nn->numLayers - 1; ++l) {
    for (int k = 0; k < nn->layers[l].size; ++k) {
      free(nn->connections[l][k]);
    }
    free(nn->connections[l]);
  }
  free(nn->connections);

  // then free neurons + layer
  for (int i = 0; i < nn->numLayers; ++i) {
    free(nn->layers[i].neurons);
  }
  free(nn->layers);

  free(nn);
}

float ReLU(float a) {
  if (a > 0.)
    return a;
  return 0.;
}

int main(void) {
  NeuralNet *nn = initNerualNet(10, 15, 20, 3, &ReLU);

  freeNeuralNet(nn);

  return EXIT_SUCCESS;
}
