cmake_minimum_required(VERSION 3.25)

project("AIModernCpp"
    VERSION 1.0.0
    DESCRIPTION "AI-Powered Application in Modern C++"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 只启用 ISO C 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# 判断操作系统
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Linux platform: ${CMAKE_SYSTEM_NAME}")
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "Windows platform: ${CMAKE_SYSTEM_NAME}")
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "Darwin platform: ${CMAKE_SYSTEM_NAME}")
else()
    message(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")
endif()

# 设置输出目录（使用生成器表达式保持跨平台）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

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
    add_compile_options(/MP)
endif()

# ------------------------------------------------------
# 设置一系列的变量, 顶层 CMake 的变量可以被底层 CMake 访问
set(NLOHMANN_JSON_VERSION "v3.12.0")
set(HTTPLIB_VERSION "v0.63.0")

# -------------------------------------------
# external 第三方库

# ---- 本地 OpenSSL 配置（必须在 httplib 之前）----
# OpenSLL on Windows download link:
# https://slproweb.com/products/Win32OpenSSL.html
set(OPENSSL_ROOT_DIR    "D:/DevelopTools/OpenSSL-Win64")
set(OPENSSL_INCLUDE_DIR "D:/DevelopTools/OpenSSL-Win64/include")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPENSSL_SSL_LIBRARY    "D:/DevelopTools/OpenSSL-Win64/lib/VC/x64/MDd/libssl.lib")
    set(OPENSSL_CRYPTO_LIBRARY "D:/DevelopTools/OpenSSL-Win64/lib/VC/x64/MDd/libcrypto.lib")
else()
    set(OPENSSL_SSL_LIBRARY    "D:/DevelopTools/OpenSSL-Win64/lib/VC/x64/MD/libssl.lib")
    set(OPENSSL_CRYPTO_LIBRARY "D:/DevelopTools/OpenSSL-Win64/lib/VC/x64/MD/libcrypto.lib")
endif()
find_package(OpenSSL 4.0 REQUIRED)

set(HTTPLIB_REQUIRE_OPENSSL          ON CACHE BOOL "" FORCE)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE ON CACHE BOOL "" FORCE)

# ======================================
include(FetchContent)

# 拉取 nlohmann/json
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/${NLOHMANN_JSON_VERSION}/include.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE

    # 核心：使用 CMAKE_CURRENT_SOURCE_DIR（项目根目录）来定位绝对路径
    # DOWNLOAD_DIR：指定压缩包或 Git 仓库元数据的下载存放路径
    # SOURCE_DIR：指定源码最终解压或检出（Checkout）的路径
    # 指定 FetchContent 在后台执行微型构建（子构建）时临时文件的存放路径
    DOWNLOAD_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/external/json/download"
    SOURCE_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/external/json/src"
    SUBBUILD_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/external/json/subbuild"
)
# 拉取 cpp-httplib
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG ${HTTPLIB_VERSION}
    GIT_SHALLOW TRUE

    DOWNLOAD_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/external/httplib/download"
    SOURCE_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/external/httplib/src"
    SUBBUILD_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/external/httplib/subbuild"
)
# 下载并使其可用
FetchContent_MakeAvailable(nlohmann_json)
FetchContent_MakeAvailable(httplib)

# Create interface target for json using fast fetch(zip)
if(NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE ${nlohmann_json_SOURCE_DIR}/include)
    add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
endif()

# 在这里添加 DEBUG_POSTFIX
set(CMAKE_DEBUG_POSTFIX "d")

# =========================================
add_subdirectory(src)
