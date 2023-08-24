cmake_minimum_required(VERSION 3.20)
project(
    DataHandlerTemplate
    VERSION 1.0.0
    DESCRIPTION "Data Handler Template for MNIST Database"
    LANGUAGES CXX
)

# List sources here
add_executable(
    DataHandlerTemplateBinary
    main.cpp
    src/data_handler.cpp
    src/data.cpp
)

target_compile_features(DataHandlerTemplateBinary PRIVATE cxx_std_23)
target_include_directories(DataHandlerTemplateBinary PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)