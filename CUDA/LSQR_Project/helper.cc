#include "helper.h"

double** reshape_array_1D_To_2D(double *arr, int height, int width) {
    double** reshaped = 0;
    reshaped = new double*[height];

    for(int i = 0; i < height; i++) {
        reshaped[i] = new double[width];
        for (int j = 0; j < width; j++) {
            reshaped[i][j] = arr[i * width + j];
        }
    }
    return reshaped;
}

void print_2D_array(double **arr, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << arr[i][j]<< " ";
        }
        std::cout << std::endl;
    }
}


void print_matrix_vector_dense_format(double *elements, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << elements[i] << " ";
    }
    std::cout << std::endl;
}

void print_matrix_vector_dense_format(int *elements, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << elements[i] << " ";
    }
    std::cout << std::endl;
}

std::tuple<int, int , double*> read_file(const char* path) {
   /* Function for reading data from a file
    First value of the file have to be the numbers of the rows and the second number is 
    the column number of the matrix/vector. Everything else numbers for the Matrix/Vector.
    :param: path: location of the file
    return:
    :param: height of the matrix/vector
    :param: width of the matrix/vector
    :param: Matrix/Vector array as a 1D-Array
    */

   int height;
   int width;
   double* elements;
    
    std::fstream file(path);

    if (file.is_open()) {
        int i = 0;
        int j = 0;

        double line;
        while (file >> line) {
            if (i == 0) {
                height = (int)line;
                i++;
            } else if (i == 1) {
                width = (int)line;
                i++;
            } else {
                /* <<<< -------   read Matrix/Vector  ------- >>>>> */
                if (i == 2 && j == 0) {
                    elements = new double[width * height];
                    elements[j] = line;
                    i++;
                    j++;
                } else {
                    elements[j] = line;
                    j++;
                }
            }
        }
        file.close();
    } else {
        std::cout << "file not found " << std::endl;
        exit(0);
    }

   return std::make_tuple(height, width, elements);
}

void save_file(const char* path, double* elements, int height, int width) {
    std::ofstream output;
    output.open (path);
    int j = 0;
    for(int i=0; i<height + 2;i++){
        if (i == 0) {
            output << height << std::endl;
        }else if (i == 1) {
            output << width << std::endl;
        } else {
            output << elements[j] << std::endl;
            j++;
        }
    }
    output.close();
}

bool check_hardware() {

    //TODO
}

CPUMatrix read_matrix_in_csr(const char *path) {
    int height;
    int width;
    double* elements;
    int* csrRow;
    int* csrCol;

    int elementSize = 0;
    int rowSize = 0;
    int columnSize = 0;

    std::fstream file(path);

    if (file.is_open()) {
        int i = 0;
        int j = 0;      /* ElementIdx */

        int numbersInRow = 0;
        int heightIdx = 0;
        int rowIdx = 0;
        int idx = 0;

        double line;
        while (file >> line) {
            if(i == 0) {
                height = (int)line;
                i++;

            }else if (i == 1) {
                width = (int)line;
                i++;

                elementSize = width * height;
                rowSize = width;
                columnSize = width * height;

                elements = new double[elementSize];
                csrRow = new int[rowSize];
                csrCol = new int[columnSize];

            } else {
                if(idx % width == 0) {
                    heightIdx = 0;
                }

                if(idx >= rowSize) {
                    rowSize = 2 * rowSize + 1;
                    csrRow = swapping_i_vector(csrRow, rowSize);
                }

                if (idx % height == 0) {
                    csrRow[rowIdx] = numbersInRow;
                    rowIdx++;
                }

                if (line != 0.0 ) {
                    if(j >= elementSize) {
                        elementSize = 2 * elementSize;
                        elements = swapping_d_vector(elements, elementSize);
                    }
                    elements[j] = line;

                    if(j >= columnSize) {
                        columnSize = 2 * columnSize;
                        csrCol = swapping_i_vector(csrCol, columnSize);
                    }
                    csrCol[j] = heightIdx;
                    
                    numbersInRow++;
                    j++;
                }

                heightIdx++;
                idx++;
            }
        }

        elementSize = j;
        elements = end_d_vector(elements, elementSize);

        columnSize = j;
        csrCol = end_i_vector(csrCol, columnSize);
        
        rowSize = rowIdx + 1;
        csrRow[rowIdx] = numbersInRow;
        csrRow = end_i_vector(csrRow, rowSize);

        file.close();
    } else {
        std::cout << "file not found " << std::endl;
        exit(0);
    }
    
    file.close();

    CPUMatrix matrix = matrix_alloc_sparse_cpu(height, width, elementSize, rowSize, columnSize);
    matrix.elements = elements;
    matrix.csrRow = csrRow;
    matrix.csrCol = csrCol;

    return matrix;

}

CPUMatrix read_data_in_csr(const char *path) {
    int height;
    int width;
    double* elements;
    int *csrIdx;

    int elementSize;
    int csrSize;

    std::fstream file(path);

    if (file.is_open()) {
        int i = 0;
        int j = 0;      /* ElementIdx */

        int idx = 0;

        double line;
        while (file >> line) {
            if(i == 0) {
                height = (int)line;
                i++;

            }else if (i == 1) {
                width = (int)line;
                i++;

                elementSize = width * height;
                csrSize = width;

                elements = new double[elementSize];
                csrIdx = new int[csrSize];

            } else {
                if( line != 0.0) {
                    if(j >= elementSize) {
                        elementSize = 2 * elementSize;
                        elements = swapping_d_vector(elements, elementSize);
                    }
                    elements[j] = line;
                    
                    if(j >= csrSize) {
                        csrSize = 2 * csrSize;
                        csrIdx = swapping_i_vector(csrIdx, csrSize);
                    }
                    csrIdx[j] = idx;
                    j++;
                }
                idx++;
            }
        }

        elementSize = j;
        elements = end_d_vector(elements, elementSize);

        csrSize = j;
        csrIdx = end_i_vector(csrIdx, csrSize);
    } else {
        std::cout << "file not found " << std::endl;
        exit(0);
    }

    for (int idx = 0; idx < elementSize; idx ++) std::cout << elements[idx] << ", ";
    std::cout << std::endl;
    for (int idx = 0; idx < csrSize; idx ++) std::cout << csrIdx[idx] << ", ";

    CPUMatrix matrix = vector_alloc_sparse_cpu(height, width, elementSize, csrSize);
    matrix.elements = elements;
    matrix.csrCol = csrIdx;

    return matrix;
}


double *swapping_d_vector(double *elements, int elementSize) {
    double *tmp = new double[elementSize];
    for (int idx = 0; idx < elementSize / 2; idx ++) tmp[idx] = elements[idx];
    delete[] elements;
    return tmp;
}

int *swapping_i_vector(int *elements, int elementSize) {
    int *tmp = new int[elementSize];
    for (int idx = 0; idx < elementSize / 2; idx ++) tmp[idx] = elements[idx];
    delete[] elements;
    return tmp;
}

double *end_d_vector(double *elements, int size) {
    double *tmp = new double[size];
    for (int idx = 0; idx < size; idx ++) tmp[idx] = elements[idx];
    return tmp;
}

int *end_i_vector(int *elements, int size) {
    int *tmp = new int[size];
    for (int idx = 0; idx < size; idx ++) tmp[idx] = elements[idx];
    return tmp;
}
