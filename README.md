# pico-ml 
Playground to try a basic machine learning model on a raspberry pi pico microcontroller 
with 2x Cortex M0+ cores @ 133Mhz, 264kB ram.

we will try out a simple pretrained fully connected model that maps
an input float to a it's sine value, ref: https://www.youtube.com/watch?v=BzzqYNYOcWc

The model was loaded into netron, and the weights were extracted as .bin files then converted to hex-text form with a hex viewer (ref. src/parser.c for model structuring)

We'll also try implementing some rudimentary 
optimizations to optimize the model running 
on the microcontroller such as:

* brute force (overclocking the cpu core)
* multicore processing
* optimization for model sparsity

### Getting Started:
* You will need a Raspberry Pi pico and a work environment preferably setup, you can follow the documentation at: https://www.raspberrypi.com/documentation/microcontrollers/c_sdk.html

* once you've downloaded the pico's SDK you can clone this repository:
```shell
$ git clone https://github.com/ZephyrLabs/pico-ml
```

* setup CMake
```shell
$ cd pico-ml/src
$ mkdir build 
$ cd build
$ cmake -DPICO_SDK_PATH=<path-to-pico-sdk> ..
```

* build the program
```shell
$ make -j
```

* hold down the BOOTSEL button on your rpi pico and plugin the USB cable to your PC to put it into DFU mode.
* copy pico-delegate.uf2 from the build folder to the rpi pico's folder

### Interacting with the device:
* The device presents a simple serial console with a baud rate of 115200, connect to 
it with the serial console app of your choice.
eg. on linux with picocom:
```shell
$ sudo picocom -b 115200 /dev/ttyACM0
```

Basic serial command guide:

* `m`: model parameters
  * `l`: load a byte of model data
  * `r`: reset model buffer
  * `p`: parse the model, verify structure

* `t`: tensor parameters:
  * `l`: load a byte of data into the tensor
  * `d`: dump the tensor buffer in the serial port, post inference
  * `r`: reset tensor buffer

* `i`: inferencing
  * `i`: run inference, if parsing was okay

eg: sending the following over serial
* load a byte into the model buffer: `ml<some data byte>`
* run inference: `ii`
* dump the tensor: `td`

### Performance stats:

* **Model Params:**
  * input_tensor_size: 4 bytes
  * output_tensor_size: 4 bytes
  * layer_count: 11
  * op_count: 66817 OPs
  * out: 0.845167


| Optimization | Exec time | Performance | Effective speedup | change in code |
|:--|---|---|---|---|
| 1. Normal | 55844 us | 1.196 MFLOPs | 1x | - |
| 2. Overclocking to 270 Mhz | 25859 us | 2.584 MFLOPs | 2.16x | `set_sys_clock_khz` at beginning of main.c |
| 3. Enabling Multicore processing | 13892 us | 4.810 MFLOPs | 4.02x (1.86x from #2) | using `gemm()` instead of `__gemm()` in parser.c under `run_model_inference() > case GEMM_OP:`
| 4. Sparsity Optimization | 6134 us | 10.893 MFLOPs | 9.10x (2.26x from #3) | using `__sgemm()` instead of any occurance `__gemm()` in blas_multicore.c
| 5. Running from memory | 4622 us | 14.456 MFLOPs | 12.08x (1.32x from #4) ⭐ | removing `const` type modifier prefix from sine_model_bin in sine_model.h |

Total optimization: `12.08x`

Note: these tests 1 - 4 involved running the model from flash memory which involves latency when fetching randomly from the XIP flash, executing from memory has been much faster.