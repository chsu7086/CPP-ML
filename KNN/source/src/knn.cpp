#include <cmath> // sqrt for L2 norm
#include <limits> // for maximum value of double
#include <map>
#include <cstdint>
#include "knn.hpp"
#include "data_handler.hpp"

KNN::KNN(int val) {

    k = val;
}
KNN::KNN() {
    // nothing to do
}
KNN::~KNN() {
    // nothing to do either
}

void KNN::find_k_nearest(Data* query_point);
void KNN::set_training_data(std::vector<Data*>* vec) {
    
    training_data = vec;
}
void KNN::set_test_data(std::vector<Data*>* vec) {

    test_data = vec;
}
void KNN::set_validation_data(std::vector<Data*>* vec) {

    validation_data = vec;
}
/* change k value while not re-loading data */
void KNN::set_k(int val) {

    k = val;
}

int KNN::predict();
double KNN::calculate_distance(Data* query_point, Data* input) {

    double distance = 0.0;
#ifdef EUCLID
    for (int i = 0; i < query_point->get_feature_vector_size(); ++i) {
        continue;
    }
#elif defined MANHATTAN
    // put Manhattan distance here
#endif
}
/* used to determin k */
double KNN::validate_performance();
/* used to prove that our model is repeatable */
double KNN::test_performance();
