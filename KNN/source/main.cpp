#include "data_handler.hpp"
#include "knn.hpp"

int main() {

    DataHandler* dh = new DataHandler();
    KNN* knn = new KNN();

    double current_validation_performance = 0.0;
    double best_validation_performance = 0.0;
    int best_k = 1;

    dh->read_feature_vector("../source/MNIST/train-images-idx3-ubyte");
    dh->read_feature_label("../source/MNIST/train-labels-idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    knn->set_training_data(dh->get_training_data());
    knn->set_test_data(dh->get_test_data());
    knn->set_validation_data(dh->get_validation_data());

    // try: set threshold and find the leat k for which performance exceeds it
    for (int i = 1; i < 13; ++i) {
        knn->set_k(i);
        current_validation_performance = knn->validate_performance();
        
        if (current_validation_performance > best_validation_performance) {
            best_validation_performance = current_validation_performance;
            best_k = i;
        }
    }

    knn->set_k(best_k);
    knn->test_performance();

    return 0;
}
