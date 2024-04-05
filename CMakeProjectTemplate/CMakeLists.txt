# 设置 CMake 所需要的最小版本
cmake_minimum_required(VERSION 3.20.0)

# 配置项目名称(solution in VS)以及简要信息
project(CMakeProjectTemplate
    VERSION 0.0.1
    DESCRIPTION "The Modern CMake and C++ Project Template"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX
)

# vcpkg configuration 需要根据平台进行配置
# "vcpkg help triplet" command: Available architecture triplets
# set(CMAKE_TOOLCHAIN_FILE "D:/Development/vcpkg/scripts/buildsystems/vcpkg.cmake")
# set(VCPKG_HOST_TRIPLET "x64-windows")
# set(VCPKG_TARGET_TRIPLET "x64-windows")
# https://learn.microsoft.com/en-us/vcpkg/reference/vcpkg-json
# the configuration JSON file 'vcpkg.json'

# MinGW+Clang | Ninja | CMake | VSCode + clangd + c/c++ Snippets
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_C_COMPILER clang) # clang | gcc | MSVC(cl)
# set(CMAKE_CXX_COMPILER clang++) # clang++ | g++ | | MSVC(cl)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 只启用 ISO C 标准的编译器标志, 而不使用特定编译器的扩展
# set(CMAKE_C_STANDARD 17)
# set(CMAKE_C_STANDARD_REQUIRED ON)
# set(CMAKE_C_EXTENSIONS OFF)

# 开启 CMake 支持的 ctest单元测试
enable_testing()

# ------------------------------------------------------
# 设置一系列的变量, 顶层 CMake 的变量可以被底层 CMake 访问
# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
# set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(Include_Dir ${CMAKE_SOURCE_DIR}/src/)

# set(MathLibInclude ${CMAKE_SOURCE_DIR}/src/MathLib)
# set(PrintModuleInclude ${CMAKE_SOURCE_DIR}/src/PrintModule)

# 条件判断 Debug & Release模式下的编译指令
if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0")
elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O3")
else()
    message("else:${CMAKE_BUILD_TYPE}")
    message("else:${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# 以下两条语句分别设置了Debug版本和Release版本可执行文件的输出目录,
# 一旦设置上面的属性,在任何环境下生成的可执行文件都将直接放在你所设置的目录.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin)

# 上面两条语句分别设置了Debug版本和Release版本库文件的输出目录,
# 一旦设置上面的属性,在任何环境下生成的库文件都将直接放在你所设置的目录.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib)

# -------------------------------------------
# external 第三方库
set(EIGEN3_INCLUDE_DIR "D:/Development/ExternalCppLibrary/eigen-3.4.0")
list(APPEND CMAKE_MODULE_PATH "${EIGEN3_INCLUDE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")

find_package(Eigen3 3.4 REQUIRED)

if(TARGET Eigen3::Eigen)
    message(STATUS "========= Eigen 3.4 is found =========")
endif()

# 添加 CMake 子目录, 寻找该目录下的 CMakeLists.txt
add_subdirectory(src)
add_subdirectory(test/test_PrintModule)
add_subdirectory(test/test_Math)

# --------------------------------------------------
# 把 "IMAGES" 指代的所有内容都拷贝到可执行文件目录下
# file(GLOB IMAGES
# "${PROJECT_SOURCE_DIR}/images/lena.jpg"
# "${PROJECT_SOURCE_DIR}/images/cat.jpg"
# )

# if(CMAKE_SYSTEM_NAME MATCHES "Windows")
# message(STATUS "====> On Windows")
# file(COPY ${IMAGES} DESTINATION "${EXECUTABLE_OUTPUT_PATH}/Debug")
# else() # Linux or Darwin(Mac OS)
# message(STATUS "====> On Linux or Mac")
# file(COPY ${LIBRARY_SHATED} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
# endif()
# --------------------------------------------------
