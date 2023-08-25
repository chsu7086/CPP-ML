#pragma once // The #pragma once directive serves to ignore subsequent inclusions

#include <iostream>
#include <vector>
#include <cstdint>

class Data {
	
    private:
        std::vector<uint8_t>* feature_vector; // no class at end
        uint8_t label;
        int enum_label; // A -> 1, B -> 2, etc.

	public:
		Data();
		~Data();
		void set_feature_vector(std::vector<uint8_t>*);
		void append_to_feature_vector(uint8_t);
		void set_label(uint8_t);
		void set_enumerated_label(int);

		int get_feature_vector_size();
		uint8_t get_label();
		int get_enumerated_label();

		std::vector<uint8_t>* get_feature_vector();

};
