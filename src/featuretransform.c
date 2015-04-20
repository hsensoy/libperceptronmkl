//
// Created by husnu sensoy on 11/04/15.
//

#include "featuretransform.h"
#include "mkl_vsl.h"
#include "mkl.h"

static FeatureTransformer_t __newFeatureTransform(enum FeatureTransform type) {
    FeatureTransformer_t ft = (FeatureTransformer_t) malloc(sizeof(struct FeatureTransformer_st));

    check_mem(ft);

    ft->type = type;



    return ft;

    error:
    exit(EXIT_FAILURE);
}

RBFSampler_t __newRBFSampler(long D, float sigma) {

    RBFSampler_t  rbf = (RBFSampler_t)malloc(sizeof(struct RBFSampler_st));

    check_mem(rbf);

    rbf->d = -1;
    rbf->nsample = D;
    rbf->samples = NULL;
    rbf->sigma  =sigma;
    rbf->scaler = sqrtf(1./D);


    rbf->partial_inst = NULL;
    newInitializedCPUVector(&(rbf->partial_inst), "transformed partial",D,matrixInitNone,NULL,NULL)

    rbf->partial_matrix = NULL;

    return rbf;


    error:

    exit(EXIT_FAILURE);
}

static eparseError_t __initRBFSampler(RBFSampler_t pSt, long d) {
    if (pSt->d == -1){


        pSt->d = d;


        newInitializedCPUMatrix(&(pSt->samples),"fourier samples", pSt->nsample,d,matrixInitNone,NULL,NULL);

        VSLStreamStatePtr stream;
        vslNewStream( &stream, VSL_BRNG_MT19937, 777 );

        log_info("Generating %ld gaussian vectors of %ld dimension with %f sigma",pSt->nsample , d ,pSt->sigma);
        int status =  vsRngGaussian( VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, pSt->nsample * d, pSt->samples->data, 0.0, pSt->sigma );

        if (status != VSL_STATUS_OK){
            log_info("Error in MKL random number generation %d", status);

            return eparseMKLError;
        }

        vslDeleteStream(&stream);
    }

    return eparseSucess;
}

FeatureTransformer_t newRBFSampler(long D, float sigma){

    FeatureTransformer_t ft = __newFeatureTransform(KERNAPROX_RBF_SAMPLER);

    ft->pDeriveObj = (void*) __newRBFSampler(D,sigma);

    return ft;
}

eparseError_t deleteFeatureTransformer(FeatureTransformer_t ft){
    // todo Implement this.

    return eparseSucess;
}

eparseError_t transform(FeatureTransformer_t ft, Vector_t in, Vector_t *out){
    RBFSampler_t rbf = NULL;
    switch(ft->type){
        case KERNAPROX_RBF_SAMPLER:

            rbf = (RBFSampler_t)ft->pDeriveObj;

            EPARSE_CHECK_RETURN(__initRBFSampler(rbf, in->n)    )

            EPARSE_CHECK_RETURN(prodMatrixVector(rbf->samples,false,in , rbf->partial_inst));

            newInitializedCPUVector(out, "transformed",2 * rbf->nsample, matrixInitNone,NULL,NULL)

            EPARSE_CHECK_RETURN(CosSinMatrix(rbf->partial_inst,*out));

            cblas_sscal((*out)->n,rbf->scaler ,(*out)->data, 1);

            break;
        case KERNAPROX_NONE:

            cloneVector(out, memoryCPU,in, "transformed input");

            break;

        default:
            return eparseNotImplementedYet;

    }


    return eparseSucess;


}

eparseError_t transformBatch(FeatureTransformer_t ft, Matrix_t in, Matrix_t *out){
    RBFSampler_t rbf = NULL;
    switch(ft->type){
        case KERNAPROX_RBF_SAMPLER:

            rbf = (RBFSampler_t)ft->pDeriveObj;

            EPARSE_CHECK_RETURN(__initRBFSampler(rbf, in->nrow)    )

            newInitializedMatrix(&(rbf->partial_matrix),memoryCPU,"Partial Matrix",rbf->nsample, in->ncol, matrixInitNone,NULL,NULL);

            EPARSE_CHECK_RETURN(prodMatrixMatrix(rbf->samples,false,in , rbf->partial_matrix));


            newInitializedMatrix(out,memoryCPU,"Transformed Matrix",2 * rbf->nsample, in->ncol, matrixInitNone,NULL,NULL);

            EPARSE_CHECK_RETURN(CosSinMatrix(rbf->partial_matrix,*out))

            cblas_sscal((*out)->n,rbf->scaler ,(*out)->data, 1);

            break;
        case KERNAPROX_NONE:

            cloneMatrix(out, memoryCPU,in, "transformed input");

            break;

        default:
            return eparseNotImplementedYet;

    }


    return eparseSucess;


}