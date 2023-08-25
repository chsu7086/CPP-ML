#pragma once

#include "data.hpp"
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>

class DataHandler {

    private:
        std::vector<Data*>* data_array; // all of the data (pre-split)
        std::vector<Data*>* training_data;
        std::vector<Data*>* test_data; // used after training (to calculate the performance of a model)
        std::vector<Data*>* validation_data; // used during training

        int num_classes;
        int feature_vector_size;
        std::map<uint8_t, int> class_map;

        const double TRAIN_SET_PERCENT = 0.75;
        const double TEST_SET_PERCENT = 0.20;
        const double VALID_SET_PERCENT = 0.05;

    public:
        DataHandler();
        ~DataHandler();

        // data file and label file are separated
        void read_feature_vector(std::string path);
        void read_feature_label(std::string path);
        void split_data();
        void count_classes();

        uint32_t convert_to_little_endian(const unsigned char* bytes);

        std::vector<Data*>* get_training_data();
        std::vector<Data*>* get_test_data();
        std::vector<Data*>* get_validation_data();
};
