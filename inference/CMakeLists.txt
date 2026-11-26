cmake_minimum_required(VERSION 3.14)
project(qwen3_tokenizer_wrapper LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third_party/tokenizer.cpp tokenizer_cpp)

set_target_properties(tokenizer_lib onig PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(qwen3_tokenizer SHARED tokenizer_c_api.cpp)
target_link_libraries(qwen3_tokenizer PRIVATE tokenizer_lib)
