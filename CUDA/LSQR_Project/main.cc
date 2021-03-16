#include <iostream>
#include <cstdlib>

#include "lsqr.h"


int main(int argc, char ** argv) {

    std::cout << "Hello PMPP Project - Free Matrix LSQR\n" << std::endl;

    const char *pathMatrixA;
    const char *pathVector_b;
    int max_iters;

    if (argc != 4) {
        std::cout << "Incorrect number of arguments" << std::endl;
        std::cout << "First Argument is the PATH of Matrix A" << std::endl;
        std::cout << "Second Argument is the PATH of Vector b" << std::endl;
        std::cout << "using TEST-MATRIX, TEST-VECTOR and TEST-LAMBDA from data-folder\n" << std::endl;

        pathMatrixA = "/gris/gris-f/homelv/dmarino/PMPP_Project/Data/matrix1050.txt";
        pathVector_b = "/gris/gris-f/homelv/dmarino/PMPP_Project/Data/vector1050.txt";
        max_iters = 5000;

    } else {
        pathMatrixA = argv[1];
        pathVector_b = argv[2];
        max_iters = atoi(argv[3]);
    }

    lsqr(pathMatrixA, pathVector_b, max_iters); 

    return EXIT_SUCCESS;
}