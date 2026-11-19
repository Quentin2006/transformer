
#include <nn.hpp>

int main() {
  nn::Config conf = {128, 5, 64, 32, nn::RELU};

  nn::NeuralNetwork nn(conf);

  return 0;
}
