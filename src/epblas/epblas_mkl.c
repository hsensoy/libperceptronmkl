#include "epblas.h"
#include "mkl_memman.h"
#include "mkl.h"

char* version(){
	return "MKL Enabled Embedding Parser BLAS";
}

void setParallism(int nslave){
       mkl_set_num_threads(nslave);
}

int getMaxParallism(){
   return mkl_get_max_threads();
}

bool getDynamicParallism(){
	return mkl_get_dynamic() == 1;
}


void printMatrix(const char* heading, Matrix_t m, FILE *fp) {
	
	long r, c;
	if (fp != NULL) {
		if (heading != NULL)
			fprintf(fp, "%s\n", heading);
		for (r = 0; r < m->nrow; r++) {
			for (c = 0; c < m->ncol; c++) {
				fprintf(fp, "%f", m->data[r * m->ncol + c]);
				if (c < m->ncol - 1)
					fprintf(fp, "\t");
			}
			fprintf(fp, "\n");
		}

		fprintf(fp, "\n");
	}
}

eparseError_t newMatrix(Matrix_t *mptr, memoryAllocationDevice_t device,
		const char *id, long nrow, long ncol) {
	size_t realbytes = sizeof(struct Matrix_st);
	size_t minbytes = sizeof(struct Matrix_st);
	
	if (*mptr == NULL) {
		*mptr = (Matrix_t) malloc(sizeof(struct Matrix_st));

		(*mptr)->data = NULL;

		if (*mptr == NULL)
			return eparseMemoryAllocationError;

		(*mptr)->capacity = 0;

		if (id != NULL) {
			((*mptr)->identifier) = strdup(id);
		} else {
			((*mptr)->identifier) = strdup("noname");
		}
	}

	(*mptr)->n = ncol * nrow;
	(*mptr)->ncol = ncol;
	(*mptr)->nrow = nrow;
	(*mptr)->dev = device;

	EPARSE_CHECK_RETURN(ensureMatrixCapacity(*mptr, (*mptr)->n));

	realbytes += sizeof(float) * (*mptr)->capacity;
	minbytes += sizeof(float) * (*mptr)->n;

	return eparseSucess;
}


eparseError_t ensureMatrixCapacity(Matrix_t mptr, long nentry) {

	if (mptr == NULL) {
		return eparseNullPointer;
	} else {
		if (nentry < mptr->capacity) {
			return eparseSucess;
		} else {
			long newCapacity = (long) ((nentry * 3. / 2) + 1);
			debug(
					"Growing <%s>(%ld x %ld)@%s from %ld (%s) entries to %ld (%s) (requested %ld (%s))",
					mptr->identifier, mptr->nrow, mptr->ncol,
					(mptr->dev == memoryGPU) ? "GPU" : "CPU", mptr->capacity,
					humanreadable_size(sizeof(float) * mptr->capacity),
					newCapacity,
					humanreadable_size(sizeof(float) * newCapacity), nentry,
					humanreadable_size(nentry * sizeof(float)));
			
         if (mptr->data == NULL) {
				check(mptr->dev == memoryCPU, "%s supports only CPU memory allocation", version());
				
				mptr->data = (float*)mkl_64bytes_malloc(sizeof(float) * newCapacity);
				
			} else {
				check(mptr->dev == memoryCPU, "%s supports only CPU memory allocation", version());

				mptr->data = (float*) mkl_64bytes_realloc(mptr->data, sizeof(float) * newCapacity);

			}

			if (mptr->data != NULL) {
				mptr->capacity = newCapacity;

				return eparseSucess;
			} else
				return eparseMemoryAllocationError;
		}
	}
	error:
		exit(1);
}

eparseError_t vstackMatrix(Matrix_t *m1, memoryAllocationDevice_t device,
		const char* id, Matrix_t m2, bool transposeM2, bool releaseM2) {

	if (m2 == NULL)
		return eparseNullPointer;
	else {
		long copy_bytes = sizeof(float) * m2->n;
		long offset = -10000000;

		if (*m1 == NULL) {
			offset = 0;

			if (transposeM2)
				EPARSE_CHECK_RETURN(newMatrix(m1, device, id, 0, m2->nrow))
			else
				EPARSE_CHECK_RETURN(newMatrix(m1, device, id, 0, m2->ncol))

			ensureMatrixCapacity(*m1, m2->n);

		} else {
			offset = (*m1)->n;

			if (((*m1)->ncol == m2->ncol && !transposeM2)
					|| ((*m1)->ncol == m2->nrow && transposeM2) ) {
				ensureMatrixCapacity((*m1), (*m1)->n + m2->n);
			} else {
				return eparseColumnNumberMissmatch;

			}
		}

		memcpy((*m1)->data + offset, m2->data, copy_bytes);

		if (transposeM2) {
			(*m1)->nrow += m2->ncol;
			(*m1)->ncol = m2->nrow;
		} else {
			(*m1)->nrow += m2->nrow;
			(*m1)->ncol = m2->ncol;
		}

		(*m1)->n += m2->n;
	}

	if (releaseM2)
		deleteMatrix(m2);

	return eparseSucess;
}

eparseError_t hstack(Matrix_t *m1, memoryAllocationDevice_t device, const char* id, Matrix_t m2, bool transposeM2, bool releaseM2){
	
	if(m2 == NULL)
		return eparseNullPointer;
	else{
		long copy_bytes = sizeof(float) * m2->n;
		long offset = -1000000;
		
		if(*m1 == NULL)
		{
			offset = 0;
			
			
			EPARSE_CHECK_RETURN(newMatrix(m1, device, id, 0, 0))
						
			ensureMatrixCapacity(*m1, m2->n);
		}else {
			offset = (*m1)->n;

			if (((*m1)->nrow == m2->nrow && !transposeM2)
					|| ((*m1)->nrow == m2->ncol && transposeM2) ) {
				ensureMatrixCapacity((*m1), (*m1)->n + m2->n);
			} else {
				return eparseColumnNumberMissmatch;

			}
		}
		
		memcpy((*m1)->data + offset, m2->data, copy_bytes);
		
		if (transposeM2) {
			(*m1)->nrow = m2->ncol;
			(*m1)->ncol += m2->nrow;
		} else {
			(*m1)->nrow = m2->nrow;
			(*m1)->ncol += m2->ncol;
		}

		(*m1)->n += m2->n;
		
	}
	
	if (releaseM2)
		deleteMatrix(m2);

	return eparseSucess;
	
}

eparseError_t __deleteMatrix(Matrix_t m) {
   if (m != NULL){
	 check(m->dev == memoryCPU, "%s supports only CPU memory allocation", version());

	 mkl_free(m->data);
	
	 free(m);
	 m = NULL;
   }else
      log_info("Matrix is already freed");


	return eparseSucess;
error:
	return eparseUnsupportedMemoryType;
}

eparseError_t newInitializedMatrix(Matrix_t *mptr,
		memoryAllocationDevice_t device, const char *id, long nrow, long ncol,
		matrixInitializer_t strategy, float *fix_value,
		void *stream) {

	//TODO: Implement a general elapsed time measurement.
	double begin = 0, end = 0, elapsed;


	EPARSE_CHECK_RETURN(newMatrix(mptr, device, id, nrow, ncol));

	if (strategy == matrixInitNone) {
		;
		//debug("Remember that matrix allocated contains garbage");
	} else if (strategy == matrixInitFixed) {
      if (fix_value == NULL){
         log_warn("Null pointer for fix_value");
         return eparseNullPointer;
      }

		for (int i = 0; i < nrow * ncol; i++)
			((*mptr)->data)[i] = *fix_value;
	} else if ( strategy == matrixInitCArray ){
      if (fix_value == NULL){
         log_warn("Null pointer to carray");
         return eparseNullPointer;
      }

		for (int i = 0; i < nrow * ncol; i++)
			((*mptr)->data)[i] = fix_value[i];
   
   } else{
		if (strategy != matrixInitRandom)
			log_info("Unknown initialization strategy. Failed back to random");

		// TODO: Implement random initialization for CPU based memory allocation.
		for (int i = 0; i < nrow * ncol; i++)
			((*mptr)->data)[i] = 0.0;
	}


	elapsed = end - begin;

	//debug("Allocation and initialization for %lux%lu matrix took %f sec.\n",
	//		nrow, ncol, elapsed);

	return eparseSucess;
}

eparseError_t cloneMatrix(Matrix_t *dst, memoryAllocationDevice_t device,
		const Matrix_t src, const char *new_id) {

	if (src == NULL) {
		return eparseNullPointer;
	} else {
		EPARSE_CHECK_RETURN(
				newInitializedMatrix(dst, device, (new_id == NULL) ? (src->identifier) : new_id, src->nrow, src->ncol, matrixInitNone, NULL, NULL))

		memcpy((*dst)->data, src->data, src->n * sizeof(float));

		return eparseSucess;
	}

}


/*
	TODO Fix this
*/
eparseError_t prodMatrixVector(Matrix_t A, bool tA, Vector_t x, Vector_t y){

	if (A->nrow == 0)
		return eparseSucess;
	else if (!((A->ncol == x->nrow && !tA) || (A->nrow == x->nrow && tA)))
		return eparseColumnNumberMissmatch;
	else{
	
		if (!tA){
			if(A->nrow != y->nrow){
				log_err("A(%ldx%ld) x x(%ld) does not conform with y(%ld)", A->nrow,A->ncol, x->nrow,y->nrow);
				return eparseColumnNumberMissmatch;
			}
          debug("A(%ldx%ld) x x(%ld) conform with y(%ld)", A->nrow,A->ncol, x->nrow,y->nrow);
		
		cblas_sgemv(CblasColMajor, CblasNoTrans,
					A->nrow, A->ncol,
					1., A->data, A->nrow,
					x->data, 1,0.,y->data, 1);

					}
		else{
			if(A->ncol != y->nrow){
				log_err("A(%ldx%ld)^T x x(%ld) does not conform with y(%ld)", A->nrow,A->ncol, x->nrow,y->nrow);
				return eparseColumnNumberMissmatch;
			}

          debug("A(%ldx%ld)^T x x(%ld) conform with y(%ld)", A->nrow,A->ncol, x->nrow,y->nrow);
		cblas_sgemv(CblasColMajor, CblasTrans,
					A->nrow, A->ncol,
					1., A->data, A->nrow,
					x->data, 1,0.,y->data, 1);
					}
					
		return eparseSucess;
	
	}
}

/*
	TODO Fix this.
*/
eparseError_t prodMatrixMatrix(Matrix_t A, bool tA, Matrix_t B, Matrix_t C){

	if (A->nrow == 0)
		return eparseSucess;
	else{	
		if (!tA){
			
			check(A->ncol == B->nrow && A->nrow == C->nrow && B->ncol == C->ncol, "A(%ldx%ld) x B(%ldx%ld) does not conform with C(%ldx%ld)", A->nrow,A->ncol, B->nrow, B->ncol,C->nrow,C->ncol)

			cblas_sgemm(CblasColMajor, CblasNoTrans,CblasNoTrans,
					A->nrow, B->ncol, A->ncol,
					1., A->data, A->nrow,B->data, B->nrow,0.,
					C->data, C->nrow);
		}
		else{
			check(A->nrow == B->nrow && A->ncol == C->nrow && B->ncol == C->ncol, "A(%ldx%ld)^T x B(%ldx%ld) does not conform with C(%ldx%ld)", A->nrow,A->ncol, B->nrow, B->ncol,C->nrow,C->ncol)
		
			cblas_sgemm(CblasColMajor, CblasTrans,CblasNoTrans,
					A->ncol, B->ncol, A->nrow,
					1., A->data, A->nrow,B->data, B->nrow,0.,
					C->data, C->nrow);
		}
		
					
		return eparseSucess;
	
	}
	
error:
	return eparseColumnNumberMissmatch;
	
}

eparseError_t powerMatrix(Matrix_t x, int power, Matrix_t y){
	if( !(x->nrow == y->nrow && x->ncol == y->ncol)){
			log_err( "x(%ldx%ld) and y(%ldx%ld) does not conform", x->nrow,x->ncol, y->nrow, y->ncol);
			return eparseColumnNumberMissmatch;
	}
	
	vmsPowx( x->n, x->data, power, y->data, VML_EP | VML_FTZDAZ_OFF | VML_ERRMODE_DEFAULT );
	
	return eparseSucess;
		
}

eparseError_t dot(Vector_t x, Vector_t y, float *result){
	if( !(x->nrow == y->nrow && x->ncol == 1 &&  y->ncol == 1)){
			log_err( "x(%ldx%ld) and y(%ldx%ld) does not conform", x->nrow,x->ncol, y->nrow, y->ncol);

			return eparseColumnNumberMissmatch;
	}
	

	*result = cblas_sdot(x->nrow, x->data, 1, y->data, 1);
	
	
	return eparseSucess;

}

eparseError_t mtrxcolcpy(Matrix_t *dst, memoryAllocationDevice_t device,
        const Matrix_t src, const char *new_id, long offsetcol, long ncol) { 
    if (src == NULL) {
        return eparseNullPointer;
    } else {
		check( ncol > 0, "Number of columns to be copied (ncol) should be positive");
		check( offsetcol >= 0 && offsetcol < src->ncol , "Column offset (offsetcol) should be between [0, # of columns source)");
		
        EPARSE_CHECK_RETURN(
                newInitializedMatrix(dst, device, (new_id == NULL) ? (src->identifier) : new_id, src->nrow, MIN(src->ncol,ncol), matrixInitNone, NULL, NULL))

		memcpy((*dst)->data, src->data + (offsetcol * src->nrow), MIN(src->ncol,ncol) * src->nrow * sizeof(float));

        return eparseSucess;
    }	
	
	error:
	return eparseFailOthers;
}

eparseError_t vappend(Vector_t *v, memoryAllocationDevice_t device, const char* id, float value){
	
    long copy_bytes = sizeof(float) * 1;
    long offset = 0;

    if (*v == NULL) {
        offset = 0;
			
		newVector(v, device, id, 1)
			
		float temp = value;
			
        memcpy((*v)->data, &temp, copy_bytes);
        
    } else {
    	check((*v)->dev == device, "You can not change memory type of %s from %d to %d", (*v)->identifier, (*v)->dev,  device);
    	
        offset = (*v)->n;


		(*v)->nrow += 1;
    		
        EPARSE_CHECK_RETURN(ensureMatrixCapacity((*v), (*v)->n + 1))

        
		float temp = value;
		
        memcpy((*v)->data + offset, &temp, copy_bytes);
        
    	(*v)->n += 1;
    }

    return eparseSucess;
error:
	return eparseMemoryAllocationError;	
}


