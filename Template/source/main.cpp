#include "data_handler.hpp"

int main() {

    DataHandler* dh = new DataHandler();

    dh->read_feature_vector("../source/MNIST/train-images-idx3-ubyte"); // from build directory to MNIST data
    dh->read_feature_label("../source/MNIST/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    return 0;
}
