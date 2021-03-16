#pragma once

#include <iostream>
#include "Testing/lsqrDense.h"
#include "matrix.h"
#include "math.h"
#include "helper.h"


void compare_lsqr(CPUMatrix A, CPUMatrix b, CPUMatrix result, int max_iters, double eps);

bool compare_sparse_format_array(CPUMatrix a1, CPUMatrix a2, double ebs);

bool distance_values(double Xi, double xi, double ebs);