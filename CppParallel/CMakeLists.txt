cmake_minimum_required(VERSION 3.25.0)

project(ModelCppParallel
    VERSION 1.0.1
    DESCRIPTION "The Models C++ Parallel and Concurrency"
    HOMEPAGE_URL "https://github.com/2694048168/CppParallel"
    LANGUAGES C CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
# set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# add_compile_definitions(<definition> ...)
# 将预编译参数添加到源文件的编译中，对下级子目录同样有效
# 预编译命令会添加到COMPILE_DEFINITIONS目录属性中

# 判断编译器
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message(STATUS "LLVM Clang compiler: ${CMAKE_CXX_COMPILER_ID}")
    # lots of warnings and all warnings as errors
    # add_compile_options(-Wall -Wextra -pedantic -Werror -finput-charset=utf-8 -fexec-charset=utf-8)
    # add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(STATUS "GNU GCC compiler: ${CMAKE_CXX_COMPILER_ID}")
    # lots of warnings and all warnings as errors
    # add_compile_options(-Wall -Wextra -pedantic -Werror -finput-charset=utf-8 -fexec-charset=utf-8)
    # add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(STATUS "Intel C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(STATUS "Visual Studio C++ compiler(MSVC): ${CMAKE_CXX_COMPILER_ID}")
    # add_compile_options(<option> ...)
    # option:编译选项,注意对于不同编译器,支持的选项可能不一样
    # 将编译器选项添加到当前及子目录的源文件的编译中
    # warning level 4 and all warnings as errors
    # =====If you want to set both the source character set and
    # the execution character set to UTF-8, you can use the /utf-8
    # compiler option as a shortcut. It's equivalent to
    # /source-charset:utf-8 /execution-charset:utf-8 on the command line.
    # add_compile_options(/W4 /WX /source-charset:utf-8 /execution-charset:utf-8)
    # add_compile_options(/W4 /WX /utf-8)
    # add_compile_options(/utf-8)
    add_compile_options(/arch:AVX)
endif()

# ---------- 条件判断 Debug & Release 模式下的编译指令 ----------
# if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
# elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
# else()
# endif()

# ------------------------------------------------------
# 设置一系列的变量, 顶层 CMake 的变量可以被底层 CMake 访问
# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# -------------------------------------------
# external 第三方库
set(TBB_Include "${PROJECT_SOURCE_DIR}/external/oneapi_tbb/include")
set(TBB_Lib "${PROJECT_SOURCE_DIR}/external/oneapi_tbb/lib/intel64/vc14/")
include_directories(${TBB_Include})
link_directories(${TBB_Lib})
file(GLOB DLL_FILE
"${PROJECT_SOURCE_DIR}/external/oneapi_tbb/redist/intel64/vc14/tbb12.dll"
)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "====> On Windows")
    if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
        file(COPY ${DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
    elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
        file(COPY ${DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Release")
    else()
        file(COPY ${DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
    endif()
else() # Linux or Darwin(Mac OS)
    message(STATUS "====> On Linux or Mac")
    file(COPY ${DLL_FILE} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
endif()

# google test
set(Gtest_Include "${PROJECT_SOURCE_DIR}/googletest/googletest/include")
include_directories(${Gtest_Include})
add_subdirectory(googletest)

# google benchmark
set(Benchmark_Include "${PROJECT_SOURCE_DIR}/benchmark/include")
include_directories(${Benchmark_Include})
add_subdirectory(benchmark)

# CMake Support OpenMP
find_package(OpenMP REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src/)
# --------------------------------------------------
# 添加 CMake 子目录, 寻找该目录下的 CMakeLists.txt
add_executable("00_tbb_parallel_for" src/tbb_parallel_for.cpp)
add_executable("01_work_distribution_uneven_job_length" src/work_distribution_uneven_job_length.cpp)
add_executable("02_coarse_grained" src/coarse_grained.cpp)
add_executable("03_fine_grained" src/fine_grained.cpp)
add_executable("04_dynamic_parti_static" src/dynamic_parti_static.cpp)
add_executable("05_dynamic_parti_dynamic" src/dynamic_parti_dynamic.cpp)
add_executable("05_dynamic_parti_dynamic2" src/dynamic_parti_dynamic2.cpp)
add_executable("06_false_sharing_serial" src/false_sharing_serial.cpp)
add_executable("07_false_sharing_direct" src/false_sharing_direct.cpp)
add_executable("08_false_sharing" src/false_sharing.cpp)
add_executable("09_no_sharing" src/no_sharing.cpp)
add_executable("10_double_buffer_baseline" src/double_buffer_baseline.cpp)
add_executable("11_double_buffer" src/double_buffer.cpp)
add_executable("12_guard_locks" src/guard_locks.cpp)
# add_executable("13_spin_locks" src/spin_locks.cpp)
# add_executable("14_fairness_baseline" src/fairness_baseline.cpp)
# add_executable("15_fairness" src/fairness.cpp)
add_executable("16_concurrent_queue_baseline" src/concurrent_queue_baseline.cpp)
add_executable("17_concurrent_queue" src/concurrent_queue.cpp)
add_executable("18_blocking" src/blocking.cpp)
add_executable("19_non_blocking" src/non_blocking.cpp)
add_executable("20_lock_free" src/lock_free.cpp)
add_executable("21_wait_free." src/wait_free.cpp)
add_executable("22_dot_product_baseline" src/dot_product_baseline.cpp)
target_link_libraries("22_dot_product_baseline" benchmark::benchmark)
add_executable("23_dot_product_simd" src/dot_product_simd.cpp)
target_link_libraries("23_dot_product_simd" benchmark::benchmark)
add_executable("24_dot_product_unsafe" src/dot_product_unsafe.cpp)
add_executable("25_dot_product_simd_" src/dot_product_simd_.cpp)
add_executable("26_thread_affinity" src/thread_affinity_os_scheduling.cpp)
target_link_libraries("26_thread_affinity" benchmark::benchmark)
add_executable("27_thread_affinity" src/thread_affinity_handle_set.cpp)
target_link_libraries("27_thread_affinity" benchmark::benchmark)
add_executable("28_memory_reordering" src/memory_reordering.cpp)
add_executable("29_memory_reordering" src/memory_reordering_hardware.cpp)
add_executable("30_data_race" src/data_race.cpp)
add_executable("31_openmp_baseline" src/openmp_baseline.cpp)
target_link_libraries("31_openmp_baseline" benchmark::benchmark)
add_executable("32_openmp_using" src/openmp_using.cpp)
target_link_libraries("32_openmp_using"
PUBLIC
    benchmark::benchmark
    OpenMP::OpenMP_CXX
)
include(CMakePrintHelpers)
cmake_print_properties(
TARGETS
    OpenMP::OpenMP_CXX
PROPERTIES
    INTERFACE_COMPILE_OPTIONS
    INTERFACE_INCLUDE_DIRECTORIES
    INTERFACE_LINK_LIBRARIES
)
add_executable("33_openmp_using_modern" src/openmp_using_modern.cpp)
target_link_libraries("33_openmp_using_modern"
PUBLIC
    benchmark::benchmark
    OpenMP::OpenMP_CXX
)
add_executable("34_tbb_using" src/tbb_using.cpp)
target_link_libraries("34_tbb_using"
PUBLIC
    benchmark::benchmark
)
