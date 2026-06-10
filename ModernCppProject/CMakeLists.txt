cmake_minimum_required(VERSION 4.3)

project("ModernCppProject"
    VERSION 1.0.0
    DESCRIPTION "Cpp and Qt Application Project via Modern C++"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX
)

# 获取当前时间（ISO 格式，也可自定义格式）
string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%d %H:%M:%S")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Version/version.h
)

# 查找 Doxygen 及其可选工具，如 dot (Graphviz)
find_package(Doxygen REQUIRED dot)

# 位置要求：必须在 project() 之后，
# 因为 project() 会初始化 CMAKE_SOURCE_DIR 等变量，
# 且可能会影响 CMAKE_INSTALL_PREFIX 的默认值检测
# ========== 新增部分：设置默认安装路径 ==========
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install"
        CACHE PATH "Default install prefix" FORCE)
    message(STATUS "CMAKE_INSTALL_PREFIX not set by user, using default: ${CMAKE_INSTALL_PREFIX}")
endif()
# =============================================

# vcpkg configuration 需要根据平台进行配置
# "vcpkg help triplet" command: Available architecture triplets
# set(CMAKE_TOOLCHAIN_FILE "D:/Development/vcpkg/scripts/buildsystems/vcpkg.cmake")
# set(VCPKG_HOST_TRIPLET "x64-windows")
# set(VCPKG_TARGET_TRIPLET "x64-windows")
# https://learn.microsoft.com/en-us/vcpkg/reference/vcpkg-json
# the configuration JSON file 'vcpkg.json'
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

# -------------------------------------------
# external 第三方库
set(EIGEN3_INCLUDE_DIR "D:/Development/ExternalCppLibrary/eigen-3.4.0")
list(APPEND CMAKE_MODULE_PATH "${EIGEN3_INCLUDE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")

set(QT_ROOT "D:/DevelopTools/QT/5.15.2/msvc2019_64/")
# 查找 Qt（使用 COMPONENTS 指定所需模块）
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)
# Enable Qt auto features
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# 在这里添加 DEBUG_POSTFIX
set(CMAKE_DEBUG_POSTFIX "d")

# =========================================
add_subdirectory(src)

# # ===== 自动化文档输出 ======================
# # ===== 配置 Doxygen 文档 ======================
# set(HHC_LOCATION "D:/DevelopTools/Doxygen/HTML_Help/hhc.exe")
# file(TO_CMAKE_PATH "${HHC_LOCATION}" HHC_LOCATION)

# # 处理 Doxyfile.in，生成 Doxyfile
# configure_file(
#     ${PROJECT_SOURCE_DIR}/Doxyfile.in
#     ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#     @ONLY
# )

# option(BUILD_DOCS "Build documentation" ON)

# if(BUILD_DOCS AND DOXYGEN_FOUND)
#     doxygen_add_docs(docs
#         # 不再需要列出源文件，因为 Doxyfile 中已经定义了 INPUT
#         Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
#         COMMENT "Generate documentation with Doxygen"
#         ALL
#     )
#     message(STATUS "Doxygen documentation will be built (including CHM).")
# else()
#     message(STATUS "Doxygen not found. Documentation will NOT be built.")
# endif()

# =========================================
# 安装目标
set(INSTALL_TARGETS 
    ModernCppApp 
    IthacaCore 
    IthacaLogger
)
install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
# 公共头文件（便于外部使用库）
install(DIRECTORY 
    src/Core/include/Core/
    DESTINATION include/Core
)
install(DIRECTORY 
    src/Logger/include/Logger/
    DESTINATION include/Logger
)
