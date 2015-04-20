#include "simpleperceptron.h"
#include "mkl.h"

eparseError_t deleteSimplePerceptron(SimplePerceptron_t sp) {

    deleteVector(sp->best_w);
    deleteVector(sp->w);
    deleteVector(sp->w_avg);

    free(sp);

    return eparseSucess;
}



SimplePerceptron_t __newSimplePerceptron() {

    SimplePerceptron_t p = (SimplePerceptron_t)malloc(sizeof(struct SimplePerceptron_st));

    check_mem(p);


    p->best_numit = 0;
    p->best_w = NULL;
    p->w = NULL;
    p->w_avg = NULL;
    p->w_beta = NULL;
    p->c=1;

    return p;




    error:
        exit(EXIT_FAILURE);
}

eparseError_t scoreSimplePerceptron(SimplePerceptron_t kp, Vector_t inst, bool avg, float *s) {
    if(avg){
        if (kp->w_avg == NULL)
            *s = 0.f;
        else{
        EPARSE_CHECK_RETURN(dot(kp->w_avg,inst,s))
        }
    }else{
        if (kp->w == NULL)
            *s = 0.f;
        else {
            EPARSE_CHECK_RETURN(dot(kp->w, inst, s))
        }
    }
}



eparseError_t scoreBatchSimplePerceptron(SimplePerceptron_t kp, Matrix_t instarr, bool avg, Vector_t *result) {
    float zero = 0.f;


    newInitializedCPUVector(result,"result",instarr->ncol, matrixInitNone,NULL,NULL)

    if(avg){
        if (kp->w_avg == NULL )
            newInitializedCPUVector(result,"result",instarr->ncol, matrixInitFixed,&zero,NULL)
        else{
            EPARSE_CHECK_RETURN(prodMatrixVector(instarr,true, kp->w_avg,*result))
        }
    }else{
        if (kp->w == NULL){

            newInitializedCPUVector(result,"result",instarr->ncol, matrixInitFixed,&zero,NULL)

            log_info("Initialized");
        }
        else {
            EPARSE_CHECK_RETURN(prodMatrixVector(instarr,true, kp->w,*result))
        }
    }

    return eparseSucess;
}

eparseError_t updateSimplePerceptron(SimplePerceptron_t kp, Vector_t sv, long svidx, float change) {
    float zero = 0.f;


    if (kp->w == NULL) {

        newInitializedCPUVector(&(kp->w),"weight",sv->n,matrixInitFixed,&zero,NULL);

        printMatrix("w",kp->w,stdout);
        newInitializedCPUVector(&(kp->w_avg),"avg weight",sv->n,matrixInitFixed,&zero,NULL);
        newInitializedCPUVector(&(kp->w_beta),"weight-beta",sv->n,matrixInitFixed,&zero,NULL);

        log_info("w,w-avg and w-beta all intialized as 0-vector of %ld length",sv->n);
    }

    cblas_saxpy(sv->n,change,sv->data,1,kp->w->data,1);
    cblas_saxpy(sv->n,change * kp->c,sv->data,1,kp->w_beta->data,1);

    return eparseSucess;

}

eparseError_t dumpSimplePerceptron(FILE *fp, SimplePerceptron_t kp) {
    // TODO: Implement
    return eparseColumnNumberMissmatch;
}

eparseError_t loadSimplePerceptron(FILE *fp, void **kp) {
    // TODO: Implement
    return eparseColumnNumberMissmatch;
}

eparseError_t recomputeSimplePerceptronAvgWeight(SimplePerceptron_t p){
    return eparseColumnNumberMissmatch;
}

eparseError_t snapshotBestSimplePerceptron(SimplePerceptron_t sp){


    return eparseColumnNumberMissmatch;
}

eparseError_t showStatsSimplePerceptron(SimplePerceptron_t sp){

    return eparseNotImplementedYet;
}