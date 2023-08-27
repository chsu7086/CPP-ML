#pragma once

#include "data.hpp"
#include <vector>
#include <cmath> // sqrt for L2 norm
#include <limits> // for maximum value of double
#include <map>
#include <cstdint>

class KNN {

    private:
    int k;
    std::vector<Data*>* neighbors; // temporarily contains k nearest neighbors
    std::vector<Data*>* training_data;
    std::vector<Data*>* test_data;
    std::vector<Data*>* validation_data;

    public:
    KNN(int val); // manually sets k value
    KNN(); // default constructor
    ~KNN();

    void find_k_nearest(Data* query_point);
    void set_training_data(std::vector<Data*>* vec);
    void set_test_data(std::vector<Data*>* vec);
    void set_validation_data(std::vector<Data*>* vec);
    void set_k(int val); // change k value while not re-loading data
    void test_performance(); // used to prove that our model is repeatable

    uint8_t predict();
    double calculate_distance(Data* query_point, Data* input);
    double validate_performance(); // used to determin k
};
