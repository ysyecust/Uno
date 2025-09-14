# CMakeLists.txt
# 编译热力学优化示例

cmake_minimum_required(VERSION 3.10)
project(ThermodynamicOptimization)

set(CMAKE_CXX_STANDARD 17)

# 找到 Uno 库
find_library(UNO_LIB uno PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../build)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..  # Uno 根目录
    ${CMAKE_CURRENT_SOURCE_DIR}/../../uno  # uno 源码目录
)

# C++ 接口示例（可运行）
add_executable(minimal_test minimal_test.cpp)
target_link_libraries(minimal_test 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../build/libuno.a
    /opt/homebrew/Cellar/highs/1.11.0/lib/libhighs.dylib
    ${CMAKE_DL_LIBS}
)

# 如果要编译 AMPL 外部函数库
# 使用 C 而不是 C++ 来避免头文件冲突
add_library(thermo_functions SHARED thermo_functions.c)
set_target_properties(thermo_functions PROPERTIES 
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
)
target_include_directories(thermo_functions PRIVATE 
    /Users/shaoyiyang/Documents/Code/amplsolver
)
target_link_libraries(thermo_functions 
    /Users/shaoyiyang/Documents/Code/amplsolver/amplsolver.a
    m  # math library
)