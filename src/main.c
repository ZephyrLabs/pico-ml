/**
 * @file main.c Main Source file for Pico Delegate
 * @author Sravan Senthilnathan
 * @brief 
 * @version 0.1
 * @date 2024-04-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "main.h"
#include "parser.h"

// example model
#include "sine_model.h"

#define LED_PIN 25

int main() {
  // overclock CPU to 270Mhz
  set_sys_clock_khz(270000, true);
  sleep_ms(100);

  gpio_init(LED_PIN);
  gpio_set_dir(LED_PIN, GPIO_OUT);

  multicore_launch_core1(blas_subroutine);

  stdio_init_all();

  while(1){
    data = getc(stdin);

    switch(data){
      case 'm':
        data = getc(stdin);
        
        switch(data){
          case 'l':
            model_buf[m_idx] = getc(stdin);
            m_idx++;
            break;

          case 'r':
            m_idx = 0;
            parse_status = 1;
            gpio_put(LED_PIN, 0);
            break;
          
          case 'p':
            parse_status = get_model_info(sine_model_bin,
                                          &input_tensor_size,
                                          &output_tensor_size,
                                          &layer_count,
                                          &op_count);

            if(parse_status == 0){
              gpio_put(LED_PIN, 1);
              parse_info.input_tensor_size = input_tensor_size;
              parse_info.output_tensor_size = output_tensor_size;
              parse_info.layer_count = layer_count;
              parse_info.op_count = op_count;                          
              printf("input_tensor_size: %d bytes\n", input_tensor_size);
              printf("output_tensor_size: %d bytes\n", output_tensor_size);
              printf("layer_count: %d\n", layer_count);
              printf("op_count: %d OPs\n", op_count);
            }
            break;
        }

        break;

      case 't':
        data = getc(stdin);
        
        switch(data){
          case 'l':
            mem_buf[t_idx] = getc(stdin);
            t_idx++;
            break;

          case 'd':
            for(int idx = 0; idx < t_idx; idx++){
              putc(mem_buf[idx], stdout);
            }
            break;

          case 'r':
            memset(mem_buf, 0, 1024);
            t_idx = 0;

          break;
      }

      case 'i':
        data = getc(stdin);

        switch(data){
          case 'i':
            if(parse_status != 0) break;

            *(float*)mem_buf = 1.0f; // comment this to override this default model input

            st = micros();
            run_model_inference(sine_model_bin, //change sine_model_bin to model_buf to run models from memory
                                mem_buf);
            en = micros();
  
            printf("out: %f\n", *(float*)mem_buf); // also only for this example
            inference_info.exec_time = en - st;
            printf("exec_time: %d us\n", en - st);
            printf("effective op/s: %.3f MOPs\n", (float)parse_info.op_count / (float)(en - st));
            
            break;
        }
    }
  }
}
