#include "Eigen/Core"
#include <iostream>
#include <nn.hpp>

int main() {
  nn::Config conf = {128, 5, 64, 32, nn::RELU};

  nn::NeuralNetwork nn(conf);

  nn::Vector input = nn::Vector::Random(conf.inputDim);

  const auto &res = nn.forward(input);

  std::cout << res.back().activations.transpose() << "\n";
  return 0;
}
