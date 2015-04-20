#include "eputil.h"

const char* eparseErrorString[] = {"Success", 
									"Undocumented Error",
									"Memory allocation error occurred", 
									"Number of columns in two matrix do not match", 
									"Null pointer",
									"Unsupported Memory Type (Only libperceptroncuda supports CUDA memory allocation and only libperceptronmkl supports MKL aligned memory allocation)",
									"Invalid operation request. Remember that some calls might be invalid for CUDA. Such as setParallism() and  getMaxParallism()",
									"Index out of bound",
									"Unsupported Kernel Type",
									"Error in loading Kernel Perceptron model from file"
									"Error in dumping Kernel Perceptron model into file.",
									"Cuda operation can not fit into available GPU memory",
									"Feature is not available yet",
								   "Intel MKL Error: Check MKL error string for more details."
								};
								
const char* eparseGetErrorString(eparseError_t status) {
	return eparseErrorString[status];
}