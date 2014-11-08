/*
 * File:   newcunittest1.c
 * Author: husnusensoy
 *
 * Created on Oct 10, 2014, 9:19:50 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include <CUnit/Basic.h>
#include "perceptron.h"

/*
 * CUnit Test Suite
 */
static int nupdate;

int init_suite(void) {
    nupdate = 20;

    return 0;
}

int clean_suite(void) {
    return 0;
}

void creatndDrop() {
    Perceptron_t pkp = newPolynomialKernelPerceptron(4, 1.);

    EPARSE_CHECK_RETURN(deletePerceptron(pkp))
}

void succesiveUpdatandScore() {
    Perceptron_t pkp = newPolynomialKernelPerceptron(2, 1.);


    Vector_t v = NULL;

    float somevalue = 1.;
    newInitializedCPUVector(&v, "vector", nupdate, matrixInitFixed, &somevalue, NULL);

    log_info("Allocation is done");

    for (int i = 0; i < nupdate; i++) {

        float result;

        EPARSE_CHECK_RETURN(score(pkp,v,false,&result))

        printf("Score %d: %f\n", i, result);


        EPARSE_CHECK_RETURN(update(pkp,v,i,1))

        check(((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->nrow == (i+1), "Expected number of sv %ld violates the truth %ld",((KernelPerceptron_t)pkp->pDeriveObj)->kernel->matrix->nrow, i )

    }

    EPARSE_CHECK_RETURN(deletePerceptron(pkp))

        return;

    error:
        exit(EXIT_FAILURE);
}

int main() {
    CU_pSuite pSuite = NULL;

    /* Initialize the CUnit test registry */
    if (CUE_SUCCESS != CU_initialize_registry())
        return CU_get_error();

    /* Add a suite to the registry */
    pSuite = CU_add_suite("Basic Test Suit", init_suite, clean_suite);
    if (NULL == pSuite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* Add the tests to the suite */
    if ((NULL == CU_add_test(pSuite, "create and drop", creatndDrop)) ||
            (NULL == CU_add_test(pSuite, "successive update and score", succesiveUpdatandScore))) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* Run all tests using the CUnit Basic interface */
    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return CU_get_error();
}
