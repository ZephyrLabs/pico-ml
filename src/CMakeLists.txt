cmake_minimum_required(VERSION 3.12)

include(pico_sdk_import.cmake)

project(delegate)

pico_sdk_init()

add_executable(delegate
    blas.c
    blas_multicore.c
    parser.c
    main.c
)

target_link_libraries(delegate pico_stdlib pico_multicore)

# enable usb output, disable uart output
pico_enable_stdio_usb(delegate 1)
pico_enable_stdio_uart(delegate 0)

pico_add_extra_outputs(delegate)
