#include "testing.h"
#include <chrono>


void compare_lsqr(CPUMatrix A, CPUMatrix b, CPUMatrix result, int max_iters, double ebs) {
    /* This function compare the result of our lsqr function and calculate with the testing files 
    
    :param *A: matrix array 
    :param *b: vector array
    :param result: endproduct of lsqr algorithm (CUDA) (VECTOR)

    :param eps: tolerance value
    */

    double** A_2D = reshape_array_1D_To_2D(A.elements, A.height, A.width);
    
    lsqrDense solver;
    solver.SetMatrix(A_2D);
    
    solver.SetEpsilon(ebs);
    solver.SetMaximumNumberOfIterations(max_iters);

    CPUMatrix solver_result = matrix_alloc_cpu(1, b.height);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    solver.Solve(b.height, A.width, b.elements, solver_result.elements);
    auto t2 = std::chrono::high_resolution_clock::now();

	auto cpu_comp_time = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();

    std::cout <<"LSQR using CPU took: " << cpu_comp_time << "ms" << std::endl;


    std::cout << "Stopped because " << solver.GetStoppingReason() << ": " << solver.GetStoppingReasonMessage() << std::endl;
    std::cout << "Used " << solver.GetNumberOfIterationsPerformed() << " Iterations" << std::endl;
    std::cout << "Estimate of final value of norm of residuals = " << solver.GetFinalEstimateOfNormOfResiduals() << std::endl;
    std::cout << "Estimate of norm of final solution = " << solver.GetFinalEstimateOfNormOfX() << std::endl;

    bool result_bool = compare_sparse_format_array(result, solver_result, ebs);

    // if (result_bool) {
    //     std::cout << "Comparison successful ....  Result are equal with tolerance of " << ebs << std::endl; 
    // } else {
    //     std::cout << "Comparison is not successful ....  Result are not equal with tolerance of " << ebs << std::endl; 
    // }
}


bool compare_sparse_format_array(CPUMatrix a1, CPUMatrix a2, double ebs) {
    /*
        For comparing the two matrices or two vector have to be saved in a single row array
    
    :param *a1: matrix or vector array 
    :param *a2: matrix or vector array 
    :param both *a1 and a2 have to be a matrix or a vector (DO NOT MIX like a1 is a matirx and a2 is a vector) 

    :param eps: tolerance value
    */

    /* <<<------- Check length of dense array --------->>>> */

    int size_a1 = a1.height * a1.width;
    int size_a2 = a2.height * a2.width;

    if (size_a1 != size_a2) {
        std::cerr << "Exeption: Given Arrays in function compare_sparse_format are not the same !!!!!" << std::endl;
        std::cerr << "testing.cc: compare_sparse_format_array(CPUMatrix a1, CPUMatrix a2, double ebs)" << std::endl;
        exit(1);
    }
    
    int in_range = 0;
    for (int i = 0; i < size_a1; i++) {
        bool distance = distance_values(a1.elements[i], a2.elements[i], ebs);
        if (distance == true) in_range++;
    }

    if (in_range == size_a1) {
        return true;
    } else {
        return false;
    }
}


bool distance_values(double Xi, double xi, double ebs) {
    /*
        This function checks only the distance/error of the value between the Xi and xi
    */
    double distance = sqrt(pow(Xi , 2) - pow(xi, 2));
    
    if (distance <= ebs) {
        return true;
    } else {
        return false;
    }
}


