/**
 * @file parser.c
 * @author Sravan Senthilnathan
 * @brief Model Parser and Inference Engine Source File
 * @version 0.1
 * @date 2024-03-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "pico/stdlib.h"
#include "string.h"
#include "stdio.h"

#include "parser.h"

int get_model_info(const unsigned char* model,
                    int* input_tensor_size,
                    int* output_tensor_size,
                    int* layer_count,
                    int* op_count){

    /* Create a pointer */
    int* model_ptr = (int*)model;
    
    /* Read Model Header data */
    *input_tensor_size = *model_ptr++;
    *output_tensor_size = *model_ptr++;
    *layer_count = *model_ptr++;

    /* Model operator count*/
    int op = 0;

    int i = 0;
    int l, m, n;

    for(; i < *layer_count; i++){
        int op_code = *model_ptr++;
        int skip = 0;
        switch (op_code){
            case VADD_OP:
                l = *model_ptr++;
                op += l;

                for(; skip < l; skip++) model_ptr++;
                break;
            case VMUL_OP:
                l = *model_ptr++;
                op += l;

                model_ptr++;
                break;
            case GEMM_OP:
                l = *model_ptr++;
                m = *model_ptr++;
                n = *model_ptr++;
                op += (2 * l * m * n);

                for(; skip < l * m; skip++) model_ptr++;
                break;
            case RELU_OP:
                l = *model_ptr++;
                op += l;

                break;
            default:
                printf("error in layer id: %d\nop code: %d\n", i, op_code);
                return 1;         
        }
    }

    /* Update operation count */
    *op_count = op;
    
    return(0);
}


int run_model_inference(const unsigned char* model,
                        unsigned char* mem_buf){

    /* Create a pointer */
    int* model_ptr = (int*)model;
    float* tensor_buf = (float*) mem_buf;

    int input_tensor_size;
    int output_tensor_size;
    int layer_count;
    
    /* Read Model Header data */
    input_tensor_size = *model_ptr++;
    output_tensor_size = *model_ptr++;
    layer_count = *model_ptr++;

    int i = 0;
    int l, m, n;

    for(; i < layer_count; i++){
        int op_code = *model_ptr++;
        int skip = 0;
        switch (op_code){
            case VADD_OP:
                l = *model_ptr++;

                __vadd(tensor_buf, (float*)model_ptr, l);

                for(; skip < l; skip++) model_ptr++;
                break;
            case VMUL_OP:
                l = *model_ptr++;
                
                __vmul(tensor_buf, *(float*)model_ptr, l);

                model_ptr++;
                break;
            case GEMM_OP:
                l = *model_ptr++;
                m = *model_ptr++;
                n = *model_ptr++;
                
                gemm(tensor_buf + (m * n), (float*)model_ptr, tensor_buf, l, m, n);
                memmove(tensor_buf, tensor_buf + (m * n), 4 * l * n);

                for(; skip < l * m; skip++) model_ptr++;
                break;
            case RELU_OP:
                l = *model_ptr++;
                
                __relu(tensor_buf, l);

                break;
            default:
                printf("error in layer id: %d\nop code: %d\n", i, op_code);
                return 1;         
        }
    }
        
    return(0);
}
