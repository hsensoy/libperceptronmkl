#include <stdio.h>
#include <stdlib.h>
#include "epblas/epblas.h"

/*
 * CUnit Test Suite
 */

Matrix_t A=NULL;
Matrix_t B=NULL;
Matrix_t C=NULL;
Matrix_t x=NULL;
Vector_t y=NULL;
Vector_t ones;

float zero=0.,one=1.,two,three,result;


void testRectangularMatrixMatrixProductTransposeWithSizingError(){

    // Dot product 4
    newInitializedCPUMatrix(&A, "matrix A", 1000, 100, matrixInitFixed, &one, NULL);
    newInitializedCPUMatrix(&B, "matrix B", 1000, 100, matrixInitFixed, &one, NULL);
    newInitializedCPUMatrix(&C, "matrix C", 10000, 10000, matrixInitFixed, &zero, NULL);


    check(eparseColumnNumberMissmatch == prodMatrixMatrix(A,false,B, C), "error in matrix matrix mult");

    newInitializedCPUVector(&y, "vector y", 100, matrixInitFixed, &zero, NULL);
    newInitializedCPUVector(&ones, "vector ones", 1000, matrixInitFixed, &one, NULL);

    check(eparseColumnNumberMissmatch == prodMatrixVector(C, false, ones, y), "error blas L2");

    float sum;

    check(eparseColumnNumberMissmatch == dot(y, ones, &sum), "error");

	exit(EXIT_SUCCESS);
error:
	exit(EXIT_FAILURE);

}


int main() {
    testRectangularMatrixMatrixProductTransposeWithSizingError();
}
