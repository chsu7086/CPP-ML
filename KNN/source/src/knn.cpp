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

void KNN::find_k_nearest(Data* query_point) {

    neighbors = new std::vector<Data*>;
    
    double distance = 0.0;
    double min = std::numeric_limits<double>::max();
    double previous_min = min; // previous minimum
    int index = 0;

    for (int i = 0; i < k; ++i) {
        if (i == 0) {
            for (int j = 0; j < training_data->size(); ++j) {
                distance = calculate_distance(query_point, training_data->at(j));
                training_data->at(j)->set_distance(distance);

                if (distance < min) {
                    min = distance;
                    index = j;
                }
            }

            neighbors->push_back(training_data->at(index));
            previous_min = min;
            min = std::numeric_limits<double>::max();
        } else {
            for (int j = 0; j < training_data->size(); ++j) {
                distance = training_data->at(j)->get_distance();

                if (distance < min && distance > previous_min) {
                    min = distance;
                    index = j;
                }
            }

            neighbors->push_back(training_data->at(index));
            previous_min = min;
            min = std::numeric_limits<double>::max();
        }
    }
}
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

uint8_t KNN::predict() {

    std::map<uint8_t, int> class_freq;
    
    uint8_t best = 0;
    int max_count = 0;
    int val;

    for (int i = 0; i < neighbors->size(); ++i) {
        
        if ( class_freq.find(neighbors->at(i)->get_label()) == class_freq.end() ) {
            class_freq[neighbors->at(i)->get_label()] = 1;
        } else {
            ++class_freq[neighbors->at(i)->get_label()];
        }
    }

    for (auto key: class_freq) {
        val = key.second;

        if (val > max_count) {
            max_count = val;
            best = key.first;
        }
    }
    
    delete neighbors;

    return best;
}
double KNN::calculate_distance(Data* query_point, Data* input) {

    double distance = 0.0;

    if (query_point->get_feature_vector_size() != input->get_feature_vector_size()) {
        std::cout << "Error: Vector Size Mismatch" << std::endl;
        exit(1);
    }

    for (int i = 0; i < query_point->get_feature_vector_size(); ++i) {
        distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
    }
    
    distance = sqrt(distance);

    return distance;
}
/* used to determin k */
double KNN::validate_performance() {

    double performance = 0.0;
    int count = 0;
    int prediction;

    int track = 0;

    for (Data* query_point: *validation_data) {
        if ( (track+1)%10 == 0 ) {
            std::cout << "Current query point: " << track+1 << "-th" << std::endl;
            std::cout << "Current performance: " << (double) count / (track+1) * 100.0 << " %" << std::endl;
        }

        find_k_nearest(query_point);
        prediction = predict();

        if (prediction == query_point->get_label()) {
            ++count;
        }

        ++track;
    }

    performance = (double) count / validation_data->size() * 100.0;
    std::cout << "Validation performance for k=" << k << ": " << performance << " %" << std::endl;
    
    return performance;
}
/* used to prove that our model is repeatable */
void KNN::test_performance() {

    double performance = 0.0;
    int count = 0;
    int prediction;

    for (Data* query_point: *test_data) {
        find_k_nearest(query_point);
        prediction = predict();

        if (prediction == query_point->get_label()) {
            ++count;
        }
    }

    performance = (double) count / test_data->size() * 100.0;

    std::cout << "Tested performance for k=" << k << ": " << performance << "%" << std::endl;
}
