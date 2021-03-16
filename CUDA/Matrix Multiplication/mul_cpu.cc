#include "matrix.h"
#include "mul_cpu.h"
#include "common.h"

void matrix_mul_cpu(const CPUMatrix &m, const CPUMatrix &n, CPUMatrix &p) {

	for (int i = 0; i < m.height; i++) {
		for (int j = 0; j < n.width; j++) {
			float sum = 0.0;
			for (int k = 0; k < n.height; k++) {
				sum = sum + (m.elements[i * m.width + k] * n.elements[k * n.width + j]);
			}
			p.elements[i * p.width + j] = sum;
		}
	}
}