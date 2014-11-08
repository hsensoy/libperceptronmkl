//
//  memman.h
//
//

#ifndef mkl_memman_h
#define mkl_memman_h

#include <stdlib.h>
#include "debug.h"

void* mkl_64bytes_malloc(size_t bytes);
void* mkl_64bytes_realloc(void* ptr, size_t newbytes);



#endif
