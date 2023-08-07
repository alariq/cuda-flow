# cmake needs this line
cmake_minimum_required(VERSION 3.18)

project(cuda-flow LANGUAGES C CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialize the CUDA Toolkit search using CUDA_HOME if the user specified it.
if (DEFINED ENV{CUDA_HOME})
  if (${CMAKE_VERSION} VERSION_LESS 3.18.0)
    set(CUDA_TOOLKIT_ROOT_DIR  "$ENV{CUDA_HOME}" CACHE PATH "Path to CUDA Toolkit.")
  else ()  # cmake 3.17 introduced FindCUDAToolkit
    set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "Path to CUDA Toolkit.")
  endif ()
endif ()

# Locate the root directory of the CUDA Toolkit.
# cmake 3.18 introduces the CUDAToolkit_LIBRARY_ROOT variable.
# This will update CUDA_HOME to a value discovered by find_package.
if (${CMAKE_VERSION} VERSION_LESS 3.18.0)
  find_package(CUDA REQUIRED)
  set(CUDA_HOME "${CUDA_TOOLKIT_ROOT_DIR}")
  get_filename_component(CUDA_LIB_ROOT "${CUDA_cudart_static_LIBRARY}" DIRECTORY)
  find_library(CUDA_LIB NAMES cuda cuda.lib PATHS "${CUDA_LIB_ROOT}" "${CUDA_LIB_ROOT}/stubs")
  include_directories("${CUDA_INCLUDE_DIRS}")
  find_library(CUDADEVRT_LIB cudadevrt PATHS "${CUDA_LIB_ROOT}" "${CUDA_LIB_ROOT}/stubs")
else () # Else, we're using cmake versions >= 3.18.
  cmake_policy(SET CMP0074 NEW) # Use CUDAToolkit_ROOT as a cmake prefix.
  find_package(CUDAToolkit REQUIRED)
  set(CUDA_LIB "${CUDA_cuda_driver_LIBRARY}")
  include_directories("${CUDAToolkit_INCLUDE_DIRS}")
  get_filename_component(CUDA_HOME "${CUDAToolkit_BIN_DIR}" DIRECTORY)
  find_library(CUDADEVRT_LIB cudadevrt PATH "${CUDAToolkit_LIBRARY_DIR}")
endif ()
message(STATUS "Using CUDA_HOME: ${CUDA_HOME}")
message(STATUS "Using CUDA_LIB:  ${CUDA_LIB}")
if (("${CUDA_HOME}" STREQUAL "") OR ("${CUDA_LIB}" STREQUAL ""))
  message(FATAL_ERROR "Failed to locate paths to the CUDA toolkit and nvcc.")
endif ()

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
#add_executable(cuda-flow src/kernel.cu)
add_executable(cuda-flow src/main.cpp src/CUDA_RANSAC_Homography.cu)

# Link your application with OpenCV libraries
target_link_libraries(cuda-flow PRIVATE ${OpenCV_LIBS} ${CUDA_LIB} opencv_xfeatures2d)
