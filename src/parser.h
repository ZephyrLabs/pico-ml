/**
 * @file parser.c
 * @author Sravan Senthilnathan
 * @brief Model Parser and Inference Engine Header File
 * @version 0.1
 * @date 2024-03-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "blas.h"

/* 

Model Binary Structure:

+---------------------+ 0x0000
| Input/Output/Layers |
+---------------------+ 0x000a
|      Layer 1 Info   |
|          +          |
|       Weights       |
+---------------------+
|      Layer 2 Info   |
|          +          |
|       Weights       |
+---------------------+
|      Layer 3 Info   |
|          +          |
|       Weights       |
+---------------------+
|      Layer 4 Info   |
|          +          |
|       Weights       |
+---------------------+

Graph Layer Info:

vadd: 1
<op: int32><dim 1: int32>

vmul: 2
<op: int32><dim 1: int32>

gemm: 3
<op: int32><dim 1: int32><dim 2: int32><dim 3: int32>

vrelu: 4
<op: int32><dim 1: int32>

*/

/**
 * @brief Extract model details and features
 * 
 * @param model pointer to model data
 * @param input_tensor_size input tensor size (in bytes)
 * @param output_tensor_size output tensor size (in bytes)
 * @param layer_count layers in the model
 * @param op_count number of multiply, add operations involved in inference
 * @return int status 0: parsing ok, 1: parsing error
 */
int get_model_info(const unsigned char* model,
                    int* input_tensor_size,
                    int* output_tensor_size,
                    int* layer_count,
                    int* op_count);

/**
 * @brief Run model inference
 * 
 * @param model pointer to model data
 * @param mem_buf pointer to working memory space for tensors
 * @return int status 0: inference ok, 1: inference error
 */
int run_model_inference(const unsigned char* model,
                        unsigned char* mem_buf);
