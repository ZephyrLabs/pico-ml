/**
 * @file main.h Main Source Header file for Pico Delegate
 * @author Sravan Senthilnathan
 * @brief 
 * @version 0.1
 * @date 2024-04-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "pico/stdlib.h"
#include "pico/multicore.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define micros() to_us_since_boot(get_absolute_time());

/**
 * Buffer Allocation for Model data and Tensor Data
 * model data -> up to 128 kbytes
 * Working Tensor Data -> 1 kbytes
**/
unsigned char model_buf[1024];
unsigned char mem_buf[1024];

/**
 * Misc data variables
 */
char data;
int m_idx = 0;
int t_idx = 0;

/**
 * Model Parameter data
 */
int input_tensor_size;
int output_tensor_size;
int layer_count;
int op_count;

/**
 * inference execution diagnostic data
 */
__uint32_t st;
__uint32_t en;

/**
 * Status Structures 
 */

int parse_status = 1;

struct parse_info{
    int input_tensor_size;
    int output_tensor_size;
    int layer_count;
    int op_count;
} parse_info;

struct inference_info{
    __uint32_t exec_time;
} inference_info;
