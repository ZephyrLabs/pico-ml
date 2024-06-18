/**
 * @file blas_multicore.h
 * @author Sravan Senthilnathan
 * @brief Multicore Accelerated BLAS Library Header File
 * @version 0.1
 * @date 2024-03-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */


#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "blas.h"
#include <stdio.h>

void blas_subroutine(){
	int32_t l, m, n, i;
	float* _x;
	float* _y;
	float* _z;
	float x;

	while(1){
		// get the opcode and fetch parameters from the FIFO accordingly
		int op_code = multicore_fifo_pop_blocking();

		switch(op_code){
			case VADD_OP:
				_y = (float*)multicore_fifo_pop_blocking();
				_x = (float*)multicore_fifo_pop_blocking();
				n = multicore_fifo_pop_blocking();

				__vadd(_y, _x, n);
				break;

			case VMUL_OP:
				_y = (float*)multicore_fifo_pop_blocking();
				x = multicore_fifo_pop_blocking();
				n = multicore_fifo_pop_blocking();

				__vmul(_y, x, n);
				break;

			case GEMM_OP:
				_z = (float*)multicore_fifo_pop_blocking();
				_x = (float*)multicore_fifo_pop_blocking();
				_y = (float*)multicore_fifo_pop_blocking();

				i = multicore_fifo_pop_blocking();
				l = multicore_fifo_pop_blocking();
				m = multicore_fifo_pop_blocking();
				n = multicore_fifo_pop_blocking();
				
				__sgemm(_z, _x, _y, i, l, m, n);
				break;
			
			case RELU_OP:
				_y = (float*)multicore_fifo_pop_blocking();
				n = multicore_fifo_pop_blocking();

				__relu(_y, n);
				break;
		}
	}
}

void vadd(float* y, float* x, int32_t n){
	int32_t batch_1 = n >> 1;
  	int32_t batch_0 = n - batch_1;

	multicore_fifo_push_blocking(VADD_OP);
	multicore_fifo_push_blocking((int32_t)(y + batch_0));
	multicore_fifo_push_blocking((int32_t)(x + batch_0));
	multicore_fifo_push_blocking(batch_1);

	__vadd(y, x, batch_0);
}

void vmul(float* y, float x, int32_t n){
	int32_t batch_1 = n >> 1;
  	int32_t batch_0 = n - batch_1;

	multicore_fifo_push_blocking(VMUL_OP);
	multicore_fifo_push_blocking((int32_t)(y + batch_0));
	multicore_fifo_push_blocking((int32_t)x);
	multicore_fifo_push_blocking(batch_1);

	__vmul(y, x, batch_0);
}

void gemm(float* z, float* x, float* y, int32_t l, int32_t m, int32_t n){
	int32_t batch_1 = l >> 1;
  	int32_t batch_0 = l - batch_1;

	multicore_fifo_push_blocking(GEMM_OP);
	multicore_fifo_push_blocking((int32_t)(z));
	multicore_fifo_push_blocking((int32_t)(x));
	multicore_fifo_push_blocking((int32_t)(y));
	multicore_fifo_push_blocking(batch_0);
	multicore_fifo_push_blocking(l);
	multicore_fifo_push_blocking(m);
	multicore_fifo_push_blocking(n);

	__sgemm(z, x, y, 0, batch_0, m, n);
}

void relu(float* y, int32_t n){
	int32_t batch_1 = n >> 1;
  	int32_t batch_0 = n - batch_1;

	multicore_fifo_push_blocking(RELU_OP);
	multicore_fifo_push_blocking((int32_t)(y + batch_0));
	multicore_fifo_push_blocking(batch_1);

	__relu(y, batch_0);
}
