#include "data_handler.hpp"

int main(int argc, char ** argv)
{
    DataHandler *dh = new DataHandler();
    dh->readInputData("data/train-images-idx3-ubyte");
    dh->readLabelData("data/train-labels-idx1-ubyte");
    dh->splitData();
    dh->countClasses();

    // g++ -std=c++17 -I./include/ -o main ./src/* extract_transform_load.cpp
    return 0;
}