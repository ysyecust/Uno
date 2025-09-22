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

# 定义公共库列表
set(UNO_LIBRARIES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../build/libuno.a
    /opt/homebrew/Cellar/highs/1.11.0/lib/libhighs.dylib
    # MUMPS 库
    /Users/shaoyiyang/Downloads/MUMPS_5.8.1/lib/libdmumps.a
    /Users/shaoyiyang/Downloads/MUMPS_5.8.1/lib/libmumps_common.a
    /Users/shaoyiyang/Downloads/MUMPS_5.8.1/lib/libpord.a
    /Users/shaoyiyang/Downloads/MUMPS_5.8.1/libseq/libmpiseq.a
    # SCOTCH 库
    /opt/homebrew/lib/libscotch.dylib
    /opt/homebrew/lib/libscotcherr.dylib
    /opt/homebrew/lib/libesmumps.dylib
    # Fortran 运行时库
    /opt/homebrew/lib/gcc/current/libgfortran.dylib
    # OpenMP
    /opt/homebrew/opt/libomp/lib/libomp.dylib
    # BLAS/LAPACK
    -framework Accelerate
    ${CMAKE_DL_LIBS}
)

# C++ 接口示例（可运行）
add_executable(minimal_test minimal_test.cpp)
target_link_libraries(minimal_test ${UNO_LIBRARIES})

# 教程示例
add_executable(tutorial_example tutorial_example.cpp)
target_link_libraries(tutorial_example ${UNO_LIBRARIES})

# 简化教程
add_executable(simple_tutorial simple_tutorial.cpp)
target_link_libraries(simple_tutorial ${UNO_LIBRARIES})

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