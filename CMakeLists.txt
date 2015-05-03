cmake_minimum_required(VERSION 2.8.4)

SET( PROJ_NAME      "perceptronmkl" )
SET( PROJ_PATH      ${CMAKE_SOURCE_DIR} )
SET( PROJ_OUT_PATH  ${CMAKE_BINARY_DIR} )
FILE( GLOB_RECURSE PROJ_SOURCES src/*.c src/epblas/*.c )
FILE( GLOB_RECURSE PROJ_HEADERS src/*.h src/epblas/*.h )
SET( PROJ_LIBRARIES "" )

SET( PROJ_INCLUDES  "include" "include/epblas" "/opt/intel/mkl/include" "/usr/local/include/shared")

SET ( MKLROOT "/opt/intel" )

#SET( PROJ_INCLUDES  "include"  "/opt/intel/composer_xe_2013_sp1.4.201/compiler/include"  "${MKLROOT}/mkl/include")

SET( CMAKE_SHARED_LINKER_FLAGS "-O3 -L${MKLROOT}/composer_xe_2013_sp1.4.201/compiler/lib -L${MKLROOT}/mkl/lib ${MKLROOT}/mkl/lib/libmkl_intel_ilp64.a ${MKLROOT}/mkl/lib/libmkl_core.a ${MKLROOT}/mkl/lib/libmkl_intel_thread.a /usr/local/lib/shared/libshared.a -liomp5 -ldl -lpthread -lm" CACHE STRING "" FORCE)

#SET( CMAKE_SHARED_LINKER_FLAGS "-O3 Projects/ai-parse/epblas/build -lepblasmkl" CACHE STRING "" FORCE )
SET( CMAKE_C_FLAGS "-std=gnu99 -O3 -g -DMKL_ILP64 -m64 -DNDEBUG" CACHE STRING "" FORCE)


PROJECT(${PROJ_NAME})
SET (${PROJ_NAME}_VERSION_MAJOR 0)
SET (${PROJ_NAME}_VERSION_MINOR 1)


INCLUDE_DIRECTORIES( ${PROJ_INCLUDES} )

ADD_LIBRARY( ${PROJ_NAME} SHARED ${PROJ_SOURCES})


#LINK_DIRECTORIES( "/Users/husnusensoy/NetBeansProjects/ai-parse/epblas/build" )

TARGET_LINK_LIBRARIES( ${PROJ_NAME} ${PROJ_LIBRARIES} )


enable_testing()

SET ( PROJ_TESTS
   "t1"
   "t2"
   "t3"
   "tMatrixVectorProductDimMismatch"
   "tMatrixVectorProductwithTranspose"
   "tRectangularMatrixMatrixProduct"
   "tRectangularMatrixMatrixProductTranspose"
   "tRectangularMatrixMatrixProductTransposeWithSizingError"
   "tSquareMatrixMatrixProduct"
   "tdotprod"
   "tmatrixvectorprod"
    "tSimplePerceptronwithRBFSampler"
    "tPerceptronUpdate"
    "tPerceptronUpdatewithRBFSampler"
    "tPerceptronBatchScorewithRBFSampler"
    "tPerceptronBatchScorewithRBFSamplerToyProblem"
"tvappend")

foreach( t ${PROJ_TESTS} )
       add_executable( test_${PROJ_NAME}_${t} tests/${t}.c)
           target_link_libraries( test_${PROJ_NAME}_${t} "perceptronmkl")

          target_link_libraries( test_${PROJ_NAME}_${t} "/usr/local/lib/shared/libshared.a") 
               add_test( test_${PROJ_NAME}_${t}  ./test_${PROJ_NAME}_${t} )
endforeach(t)

install(TARGETS ${PROJ_NAME} DESTINATION lib/${PROJ_NAME})
install(FILES include/perceptron_common.h include/perceptron.h include/featuretransform.h DESTINATION include/perceptron)
install(FILES include/epblas/eputil.h include/epblas/epblas.h DESTINATION include/perceptron/epblas)







