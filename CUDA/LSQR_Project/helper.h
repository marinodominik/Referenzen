#pragma once
#include <iostream>
#include <tuple>
#include <fstream>
#include "matrix.h"



double** reshape_array_1D_To_2D(double *arr, int width, int height);

void print_2D_array(double **arr, int width, int height);

void print_matrix_vector_dense_format(double *elements, int size);
void print_matrix_vector_dense_format(int* elements, int size);

std::tuple<int, int , double*> read_file(const char* path); 

void save_file(const char* path, double* elements, int height, int widht);

/* <<<<<--------- READ DATA IN COMPRESED SPARSE ROW FORMAT ----------->>>>>>>>>> */
CPUMatrix read_matrix_in_csr(const char *path);
CPUMatrix read_data_in_csr(const char *path);

double *swapping_d_vector(double *elements, int elementSize);
int *swapping_i_vector(int *elements, int elementSize);
double *end_d_vector(double *elements, int size);
int *end_i_vector(int *elements, int size);