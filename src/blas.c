/**
 * @file blas.c
 * @author Sravan Senthilnathan
 * @brief BLAS Library Source File
 * @version 0.1
 * @date 2024-03-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "pico/stdlib.h"
#include "blas.h"
#include "string.h"

void __vadd(float* y, float* x, int32_t n){
	int i = 0;
	for (; i < n; i++){
		y[i] = x[i] + y[i];
	}
}

void __vmul(float* y, float x, int32_t n){
	int i = 0;
	for (; i < n; i++){
		if(y[i] != 0)
			y[i] = y[i] * x;
	}
}

void __gemm(float* z, float* x, float* y, int32_t i, int32_t l, int32_t m, int32_t n){
	for (; i < l; i++) {
		int j = 0;
		int iy = i * n;
		int ix = i * m;
        for (; j < n; j++) {
           	int k = 0;
			float acc = 0;
           	
			for (; k < m; k++) {
				acc += x[ix + k] * y[k * n + j];
           	}

			z[iy + j] = acc;
       	}
    }
}

void __sgemm(float* z, float* x, float* y, int32_t i, int32_t l, int32_t m, int32_t n){
	for (; i < l; i++) {
		int j = 0;
		int iy = i * n;
		int ix = i * m;
        for (; j < n; j++) {
           	int k = 0;
			float acc = 0;
           	
			for (; k < m; k++) {
				float b = y[k * n + j];
				if(*(int*)&b != 0) acc += x[ix + k] * b; // sparsity hack, cast to ptr and compare as integer, cheaper, costs less cpu cycles as we don't have native float instr
           	}
			z[iy + j] = acc;
       	}
    }
}

void __relu(float* y, int32_t n){
	int i = 0;
	for (;i < n; i++){
		if(y[i] < 0.0) y[i] = (float)0x00000000; // also part of the sparsity hack, pedantically setting the float to 0
	}
}
