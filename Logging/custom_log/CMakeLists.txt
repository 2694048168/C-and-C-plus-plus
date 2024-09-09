cmake_minimum_required(VERSION 3.20.0)

project("CustomLogging"
    VERSION 1.0.1
    DESCRIPTION "Custom Logging for project via Modern C++"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_C_COMPILER clang) # clang | gcc | MSVC(cl)
# set(CMAKE_CXX_COMPILER g++) # clang++ | g++ | | MSVC(cl)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 只启用 ISO C 标准的编译器标志, 而不使用特定编译器的扩展
# set(CMAKE_C_STANDARD 18)
# set(CMAKE_C_STANDARD_REQUIRED ON)
# set(CMAKE_C_EXTENSIONS OFF)

# 判断操作系统
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Linux platform: ${CMAKE_SYSTEM_NAME}")
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "Windows platform: ${CMAKE_SYSTEM_NAME}")
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "Darwin platform: ${CMAKE_SYSTEM_NAME}")
else()
    message(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")
endif(CMAKE_SYSTEM_NAME MATCHES "Linux")

# add_compile_definitions(<definition> ...)
# 将预编译参数添加到源文件的编译中，对下级子目录同样有效
# 预编译命令会添加到COMPILE_DEFINITIONS目录属性中

# 判断编译器
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message(STATUS "LLVM Clang compiler: ${CMAKE_CXX_COMPILER_ID}")

    # lots of warnings and all warnings as errors
    # add_compile_options(-Wall -Wextra -pedantic -Werror -finput-charset=utf-8 -fexec-charset=utf-8)
    add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(STATUS "GNU GCC compiler: ${CMAKE_CXX_COMPILER_ID}")

    # lots of warnings and all warnings as errors
    # add_compile_options(-Wall -Wextra -pedantic -Werror -finput-charset=utf-8 -fexec-charset=utf-8)
    add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
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
    add_compile_options(/utf-8)
endif()

# =========================================
add_executable("test_example" main.cpp logging.cpp)
add_executable("test_benchmark" benchmarkMain.cpp logging.cpp)
