#include "data_handler.hpp"

DataHandler::DataHandler() {

    // Allocating a heap to each of the following
    data_array = new std::vector<Data*>;
    training_data = new std::vector<Data*>;
    test_data = new std::vector<Data*>;
    validation_data = new std::vector<Data*>;
}
DataHandler::~DataHandler() {

    // free dynamically allocated stuffs
}

// data file and label file are separated
void DataHandler::read_feature_vector(std::string path) {
    
    uint32_t header[4]; // |MAGIC|NUM_IMAGES|ROWSIZE|COLSIZE|
    unsigned char bytes[4]; // contains 32 bits in big endian, to be converted to little endian
    FILE* f = fopen(path.c_str(), "rb");

    if (f) {
        for (int i = 0; i < 4; ++i) {
            if (fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = convert_to_little_endian(bytes);
            } else {
                std::cout << "Error reading from file." << std::endl;
                exit(1);
            }
        }

        std::cout << "Done getting image file header." << std::endl;

        int image_size = header[2] * header[3];

        for (int i = 0; i < header[1]; ++i) {
            Data* d = new Data();

            uint8_t element[1];
            
            for (int j = 0; j < image_size; ++j) {
                if (fread(element, sizeof(element), 1, f)) {
                    d->append_to_feature_vector(element[0]);
                } else {
                    std::cout << "Error reading from file." << std::endl;
                    exit(1);
                }
            }

            data_array->push_back(d);
        }

        std::cout << "Successfully read and stored " << data_array->size() << " feature vectors." << std::endl;
    } else {
        std::cout << "Could not find the image file." << std::endl;
        exit(1);
    }
}
void DataHandler::read_feature_label(std::string path) {

    uint32_t header[2]; // |MAGIC|NUM_ITEMS|
    unsigned char bytes[4]; // contains 32 bits in big endian, to be converted to little endian
    FILE* f = fopen(path.c_str(), "rb");

    uint8_t element[1];

    if (f) {
        for (int i = 0; i < 2; ++i) {
            if (fread(bytes, sizeof(bytes), 1, f)) {
                header[i] = convert_to_little_endian(bytes);
            } else {
                std::cout << "Error reading from file." << std::endl;
                exit(1);
            }
        }

        std::cout << "Done getting label file header." << std::endl;

        for (int i = 0; i < header[1]; ++i) {
            if (fread(element, sizeof(element), 1, f)) {
                data_array->at(i)->set_label(element[0]);
            } else {
                std::cout << "Error reading from file." << std::endl;
                exit(1);
            }
        }

        std::cout << "Successfully read and stored labels." << std::endl;
    } else {
        std::cout << "Could not find the label file." << std::endl;
        exit(1);
    }
}
void DataHandler::split_data() {

	int train_size = data_array->size() * TRAIN_SET_PERCENT;
	int test_size = data_array->size() * TEST_SET_PERCENT;

	std::vector<int> rand_indices(data_array->size());
	unsigned seed = std::chrono::system_clock().now().time_since_epoch().count(); // set seed using time
	std::iota(rand_indices.begin(), rand_indices.end(), 0); // rand_indices : 0, 1, 2, ..., data_array->size()-1
	std::shuffle(rand_indices.begin(), rand_indices.end(), std::default_random_engine(seed)); // shuffle the indices

	for (int i = 0; i < data_array->size(); ++i) {
		if (i < train_size)
			training_data->push_back( data_array->at(rand_indices[i]) );
		else if (i < train_size+test_size)
			test_data->push_back( data_array->at(rand_indices[i]) );
		else
			validation_data->push_back( data_array->at(rand_indices[i]) );
	}

    std::cout << "Done splitting." << std::endl;
	std::cout << "Training Data Size: " << training_data->size() << std::endl;
	std::cout << "Test Data Size: " << test_data->size() << std::endl;
	std::cout << "Validation Data Size: " << validation_data->size() << std::endl;
}
void DataHandler::count_classes() {

    int count = 0;

    for (int i = 0; i < data_array->size(); ++i) {
        if(class_map.find(data_array->at(i)->get_label()) == class_map.end()) {
            // the current label is not in the map
            class_map[data_array->at(i)->get_label()] = count;
            data_array->at(i)->set_enumerated_label(count);
            ++count;
        } else {
            data_array->at(i)->set_enumerated_label( class_map[data_array->at(i)->get_label()] );
        }
    }

    num_classes = count;
    std::cout << "Successfully extracted " << num_classes << " unique classes." << std::endl;
}

uint32_t DataHandler::convert_to_little_endian(const unsigned char* bytes) {

    return (uint32_t) (
		(bytes[0] << 24) |
		(bytes[1] << 16) |
		(bytes[2] << 8) |
		(bytes[3])
	);
}

std::vector<Data*>* DataHandler::get_training_data() {

    return training_data;
}
std::vector<Data*>* DataHandler::get_test_data() {

    return test_data;
}
std::vector<Data*>* DataHandler::get_validation_data() {

    return validation_data;
}
