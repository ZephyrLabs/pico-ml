/**
 * @file blas.h
 * @author Sravan Senthilnathan
 * @brief BLAS Library Header File
 * @version 0.1
 * @date 2024-03-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#define VADD_OP 1
/**
 * @brief Vector Addition: eg. dst[] = src1[] + src2[]
 * 
 * @param y dst/src1 buffer
 * @param x src2 buffer
 * @param n buffer size
 */

void vadd(float* y, float* x, int32_t n);
void __vadd(float* y, float* x, int32_t n);

#define VMUL_OP 2
/**
 * @brief Vector Scaling: eg. dst = src[] * x
 * 
 * @param y dst/src buffer
 * @param x scaling factor
 * @param n buffer size
 */

void vmul(float* y, float x, int32_t n);
void __vmul(float* y, float x, int32_t n);


#define GEMM_OP 3
/**
 * @brief Matrix Multiplication: eg. dst[l][n] = src1[l][m] x src2[m][n]
 * 
 * @param c dst matrix flattened buffer
 * @param a src1 matrix flattened buffer
 * @param b src2 matrix flattened buffer
 * @param l l - matrix 1 dim
 * @param m m - matrix 1 and 2 dim
 * @param n n - matrix 2 dim
 */

void gemm(float* c, float* a, float* b, int32_t l, int32_t m, int32_t n);
void __gemm(float* c, float* a, float* b, int32_t i, int32_t l, int32_t m, int32_t n);
void __sgemm(float* c, float* a, float* b, int32_t i, int32_t l, int32_t m, int32_t n);

#define RELU_OP 4
/**
 * @brief ReLU Activation Function: eg. dst[] = src
 * @param y dst/src buffer
 * @param n buffer size
 */

void relu(float* y, int32_t n);
void __relu(float* y, int32_t n);

/**
 * @brief subroutine function for core 1
 * 
 */
void blas_subroutine();