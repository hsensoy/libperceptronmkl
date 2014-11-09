/*
 * File:   newcunittest1.c
 * Author: husnusensoy
 *
 * Created on Oct 10, 2014, 9:19:50 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <CUnit/Basic.h>
#include "epblas.h"

/*
 * CUnit Test Suite
 */

Matrix_t A;
Matrix_t B;
Matrix_t C;
Matrix_t x;
Vector_t y;
Vector_t ones;
	
float zero,one,two,three,result;

int init_suite(void) {
	 A = NULL;
	 x = NULL;
	 y = NULL;
	 ones = NULL;

	 
	
	 zero = 0;
	 one = 1;
	 two = 2;
	 three = 3;
	 
	 newInitializedCPUVector(&ones, "dummy ones", 1000, matrixInitFixed, &one, NULL);
	
	return 0;
}

int clean_suite(void) {
    return 0;
}

void testDotProduct() {
	Vector_t a = NULL;
	Vector_t b = NULL;
	
	float one = 1;
	float two = 2;
	float three = 3;
	float result ;
	
	newInitializedCPUVector(&a, "vector A", 1000, matrixInitFixed, &one, NULL);
	newInitializedCPUVector(&b, "vector B", 1000, matrixInitFixed, &one, NULL);
	
	dot(a, b, &result);
	CU_ASSERT_EQUAL(1000, result);
	
	newInitializedCPUVector(&a, "vector A", 1000, matrixInitFixed, &one, NULL);
	newInitializedCPUVector(&b, "vector B", 1000, matrixInitFixed, &two, NULL);
	
	dot(a, b, &result);
	
	CU_ASSERT_EQUAL(2000, result);
	
	newInitializedCPUVector(&a, "vector A", 1000, matrixInitFixed, &three, NULL);
	newInitializedCPUVector(&b, "vector B", 1000, matrixInitFixed, &three, NULL);
	
	EPARSE_CHECK_RETURN(dot(a, b, &result))
	
	CU_ASSERT_EQUAL(9000, result);
}

void testMatrixVectorProduct() {
	
	// Dot product 1
	newInitializedCPUMatrix(&A, "matrix A", 1000, 1000, matrixInitFixed, &one, NULL);
	newInitializedCPUVector(&x, "vector x", 1000, matrixInitFixed, &one, NULL);
	newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
	
	
	
	EPARSE_CHECK_RETURN(prodMatrixVector(A, false, x, y))
	
	float sum = 0.0;
	for(int i = 0;i < y->nrow;i++){
		sum += (y->data)[i];
	}
	
	CU_ASSERT_EQUAL(1000000, sum);
	
	// Dot product 2
	newInitializedCPUMatrix(&A, "matrix A", 1000, 1000, matrixInitFixed, &two, NULL);
	newInitializedCPUVector(&x, "vector x", 1000, matrixInitFixed, &one, NULL);
	newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
	
	
	EPARSE_CHECK_RETURN(prodMatrixVector(A, false, x, y))
	
	EPARSE_CHECK_RETURN(dot(y, ones, &sum))
	
	
	CU_ASSERT_EQUAL(2000000, sum);
	

}

void testMatrixVectorProductDimMismatch(){
	// Dot product 3
	newInitializedCPUMatrix(&A, "matrix A", 100, 1000, matrixInitFixed, &two, NULL);
	newInitializedCPUVector(&x, "vector x", 1000, matrixInitFixed, &one, NULL);
	newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
	
	
	CU_ASSERT_EQUAL(eparseColumnNumberMissmatch,prodMatrixVector(A, true, x, y))
}

void testMatrixVectorProductwithTranspose(){
	
	// Dot product 4
	newInitializedCPUMatrix(&A, "matrix A", 100, 1000, matrixInitFixed, &two, NULL);
	newInitializedCPUVector(&x, "vector x", 100, matrixInitFixed, &one, NULL);
	newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
	
	float sum;
	EPARSE_CHECK_RETURN(prodMatrixVector(A, true, x, y))
	
	EPARSE_CHECK_RETURN(dot(y, ones, &sum))
	

	CU_ASSERT_EQUAL(200000, sum);
}

void testSquareMatrixMatrixProduct(){
	
	// Dot product 4
	newInitializedCPUMatrix(&A, "matrix A", 1000, 1000, matrixInitFixed, &one, NULL);
	newInitializedCPUMatrix(&B, "matrix B", 1000, 1000, matrixInitFixed, &one, NULL);
	newInitializedCPUMatrix(&C, "matrix C", 1000, 1000, matrixInitFixed, &zero, NULL);
	
	
	EPARSE_CHECK_RETURN(prodMatrixMatrix(A,B, false, C))

	newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
	EPARSE_CHECK_RETURN(prodMatrixVector(C, false, ones, y))

	float sum;	
	EPARSE_CHECK_RETURN(dot(y, ones, &sum))
	
	CU_ASSERT_EQUAL(1000000000, sum);
}

void testRectangularMatrixMatrixProduct(){
	
	// Dot product 4
	newInitializedCPUMatrix(&A, "matrix A", 1000, 100, matrixInitFixed, &one, NULL);
	newInitializedCPUMatrix(&B, "matrix B", 100, 1000, matrixInitFixed, &one, NULL);
	newInitializedCPUMatrix(&C, "matrix C", 1000, 1000, matrixInitFixed, &zero, NULL);
	
	
	EPARSE_CHECK_RETURN(prodMatrixMatrix(A,B, false, C))

	newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
	EPARSE_CHECK_RETURN(prodMatrixVector(C, false, ones, y))

	float sum;	
	EPARSE_CHECK_RETURN(dot(y, ones, &sum))
	
	CU_ASSERT_EQUAL(100000000, sum);
}

void testRectangularMatrixMatrixProductTranspose(){
	
	// Dot product 4
	newInitializedCPUMatrix(&A, "matrix A", 1000, 100, matrixInitFixed, &one, NULL);
	newInitializedCPUMatrix(&B, "matrix B", 1000, 100, matrixInitFixed, &one, NULL);
	newInitializedCPUMatrix(&C, "matrix C", 1000, 1000, matrixInitFixed, &zero, NULL);
	
	
	EPARSE_CHECK_RETURN(prodMatrixMatrix(A,B, true, C))

	newInitializedCPUVector(&y, "vector y", 1000, matrixInitFixed, &zero, NULL);
	EPARSE_CHECK_RETURN(prodMatrixVector(C, false, ones, y))

	float sum;	
	EPARSE_CHECK_RETURN(dot(y, ones, &sum))
	
	CU_ASSERT_EQUAL(100000000, sum);
}

void testRectangularMatrixMatrixProductTransposeWithSizingError(){
	
	// Dot product 4
	newInitializedCPUMatrix(&A, "matrix A", 1000, 100, matrixInitFixed, &one, NULL);
	newInitializedCPUMatrix(&B, "matrix B", 1000, 100, matrixInitFixed, &one, NULL);
	newInitializedCPUMatrix(&C, "matrix C", 10000, 10000, matrixInitFixed, &zero, NULL);
	
	
	CU_ASSERT_EQUAL(eparseColumnNumberMissmatch, prodMatrixMatrix(A,B, true, C))
	
	newInitializedCPUVector(&y, "vector y", 100, matrixInitFixed, &zero, NULL);
	
	CU_ASSERT_EQUAL(eparseColumnNumberMissmatch, prodMatrixVector(C, false, ones, y))

	float sum;	
	
	CU_ASSERT_EQUAL(eparseColumnNumberMissmatch, dot(y, ones, &sum))
	
}

int main() {
    CU_pSuite pSuite = NULL;

    /* Initialize the CUnit test registry */
    if (CUE_SUCCESS != CU_initialize_registry())
        return CU_get_error();

    /* Add a suite to the registry */
    pSuite = CU_add_suite("EPBLAS Test Suit", init_suite, clean_suite);
    if (NULL == pSuite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* Add the tests to the suite */
    if ((NULL == CU_add_test(pSuite, "test dot product", testDotProduct)) ||
        (NULL == CU_add_test(pSuite, "test matrix vector product", testMatrixVectorProduct)) ||
    	(NULL == CU_add_test(pSuite, "test matrix vector product with transpose", testMatrixVectorProductwithTranspose)) ||
        (NULL == CU_add_test(pSuite, "test matrix vector product dim mismatch", testMatrixVectorProductDimMismatch)) ||
        (NULL == CU_add_test(pSuite, "test square matrix multiplication", testSquareMatrixMatrixProduct))||
        (NULL == CU_add_test(pSuite, "test rectengular matrix multiplication", testRectangularMatrixMatrixProduct))||
        (NULL == CU_add_test(pSuite, "test rectengular matrix multiplication with transpose", testRectangularMatrixMatrixProductTranspose))||
         (NULL == CU_add_test(pSuite, "test rectengular matrix multiplication with transpose with sizing error", testRectangularMatrixMatrixProductTransposeWithSizingError))
        
    ) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* Run all tests using the CUnit Basic interface */
    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return CU_get_error();
}
