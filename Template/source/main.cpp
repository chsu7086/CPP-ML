#include "data_handler.hpp"

int main() {

    auto dh = new DataHandler();

    dh->read_feature_vector("MNIST/train-images-idx3-ubyte");
    dh->read_feature_label("MNIST/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    return 0;
}
