#include "data_handler.hpp"

int main() {

    DataHandler* dh = new DataHandler();

    dh->read_feature_vector("/home/chsu/CPP-ML/Template/source/MNIST/train-images-idx3-ubyte");
    dh->read_feature_label("/home/chsu/CPP-ML/Template/source/MNIST/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    return 0;
}
