#pragma once

#include <Eigen/Core>
#include <functional>
#include <vector>

namespace nn {

enum ACTIVATION { RELU };

using Scalar = float;
using Vector =
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>; // nx1 mat holding scalars
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic,
                             Eigen::Dynamic>; // nxm mat holding scalars

struct Layer {
  Matrix weight; // rows = out neurons, cols = in neurons
  Vector bias;   // size = out neurons
  Vector activations;
};

struct Config {
  int inputDim, hiddenDim, outputDim, hiddenLayerCount;
  ACTIVATION activation;
};

class NeuralNetwork {
public:
  NeuralNetwork(const Config &);

private:
  static Scalar ReLU(Scalar);

  std::vector<Layer> layers;
  std::function<Scalar(Scalar)> activationFunction;
  Config config;
};

} // namespace nn
