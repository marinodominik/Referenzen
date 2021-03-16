#pragma once
#include "matrix.h"


void lsqr(const char *pathMatrixA, const char *pathVector_b, int max_iters);
void printTruncatedVector(CPUMatrix toPrint);