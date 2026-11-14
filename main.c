#include <stdlib.h>

/*
 * we can use linear algebra to compute the next layers activation
 *
 * quick recap, neural networks are made up of neurons and connections
 *
 * neurons are made up of activations and bias
 *
 * connections connect neurons from touching layers eachother with a weight
 *
 * we compure the next layers activation by multiplying the previous layers
 * activation by the connections weight and adding the bias, then passing it
 * through an activation functions(ReLU, sig, ...).
 *
 *
 *
 *
 * using linear algebra, we can compute the next layers activation via the
 * following formula:
 *
 * activation(W_l*A_l+B_l) = A_l+1
 *
 * where:
 *    W_l is the weight matrix connection layer l to layer l+1,
 *    A_l are the activation for all neurons in layer l
 *    B_l are the biasis for all neurons in layer l
 *    activation is some function that turns any float -> [0-1]
 *
 *
 *
 *
 * W_l is a kxn matrix, where
 * the rows encode the connection from all neuron in the first layer to one
 * neuron i in the next.
 *
 * W_l = w_0_0 w_0_1 ... w_0_n
 *       w_1_0 w_0_1 ... w_0_n
 *                   ...
 *       w_K_0 w_K_1 ... w_K_n
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
 * n_0_n        n_1_k
 *
 * and thats all what a neural net is
 */

int main(void) { return EXIT_SUCCESS; }
