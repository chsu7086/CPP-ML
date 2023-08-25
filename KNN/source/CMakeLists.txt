cmake_minimum_required(VERSION 3.5)
project(
    KNN
    VERSION 1.0.0
    DESCRIPTION "K-Nearest Neighbors Algorithm for MNIST Hand-written Digits"
    LANGUAGES CXX
)

# List sources here
add_executable(
    KNNBinary
    main.cpp
    src/data_handler.cpp
    src/data.cpp
    src/knn.cpp
)

target_compile_features(KNNBinary PRIVATE cxx_std_23)
target_include_directories(KNNBinary PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)