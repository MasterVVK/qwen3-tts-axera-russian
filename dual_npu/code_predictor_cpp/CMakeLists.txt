cmake_minimum_required(VERSION 3.20)
project(code_predictor_server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build with NEON
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=armv8.2-a+fp16 -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime paths
# Headers: downloaded to ./ort_include/
# Library: from pip package
set(ORT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ort_include")
set(ORT_LIB_DIR "/usr/local/lib/python3.11/dist-packages/onnxruntime/capi")

add_executable(code_predictor_server code_predictor_server.cpp)

target_include_directories(code_predictor_server PRIVATE ${ORT_INCLUDE_DIR})
target_link_directories(code_predictor_server PRIVATE ${ORT_LIB_DIR})
target_link_libraries(code_predictor_server PRIVATE onnxruntime pthread)

# Set RPATH so the binary finds libonnxruntime at runtime
set_target_properties(code_predictor_server PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR}"
    INSTALL_RPATH "${ORT_LIB_DIR}"
)
