cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flash_attn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYTHON_VENV_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv" CACHE PATH "Path to Python virtual environment")

# Specify the Python virtual environment path
set(Python_ROOT_DIR "${PYTHON_VENV_PATH}")

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Execute a Python script to find torch include and library directories
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(';'.join(cpp_extension.include_paths()), end='')"
    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0], end='')"
    OUTPUT_VARIABLE TORCH_LIBRARY_DIR
)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__version__, end='')"
    OUTPUT_VARIABLE TORCH_VERSION
)

include_directories(
    ${Python_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_attn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
)
link_directories(${TORCH_LIBRARY_DIR})

# Source files
set(SOURCES
    csrc/flash_attn/flash_api.cpp
    csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu
    csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu
    csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu
)

# Create the CUDA library
add_library(flash_attn_2_cuda SHARED ${SOURCES})

# Set CUDA architectures
set_property(TARGET flash_attn_2_cuda PROPERTY CUDA_ARCHITECTURES 80)
if(${CUDAToolkit_VERSION_MAJOR} GREATER_EQUAL 11 AND ${CUDAToolkit_VERSION_MINOR} GREATER_EQUAL 8)
    set_property(TARGET flash_attn_2_cuda PROPERTY CUDA_ARCHITECTURES 80 90)
endif()

# Set CUDA compile options
target_compile_options(flash_attn_2_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -std=c++17
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --use_fast_math
    >
)

set(TORCH_LIBRARIES torch torch_cpu torch_cuda c10 c10_cuda torch_python)

target_link_libraries(flash_attn_2_cuda PRIVATE ${TORCH_LIBRARIES} Python::Python CUDA::cudart)

set_target_properties(flash_attn_2_cuda PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/lib"
)

# Add PyTorch extension-specific flags
target_compile_definitions(flash_attn_2_cuda PRIVATE TORCH_EXTENSION_NAME=flash_attn_2_cuda)
target_compile_definitions(flash_attn_2_cuda PRIVATE TORCH_VERSION_MAJOR=${TORCH_VERSION})

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_LIBRARY_DIR}/*.dll")
    add_custom_command(TARGET flash_attn_2_cuda
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:flash_attn_2_cuda>
    )
endif()