#pragma once

#include <Eigen/Core>
#include <cstddef>
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
  Vector z;
  Vector activations;
};

struct Config {
  size_t inputDim, hiddenDim, outputDim, hiddenLayerCount;
  ACTIVATION activation;
};

class NeuralNetwork {
public:
  NeuralNetwork(const Config &);

  const std::vector<Layer> &forward(const Vector &input);

  const Config &getConf(void) { return config; }

private:
  static Scalar ReLU(Scalar);

  std::vector<Layer> layers;
  std::function<Scalar(Scalar)> activationFunction;
  Config config;
};

} // namespace nn
