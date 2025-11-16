cmake_minimum_required(VERSION 3.25.0)

project("VTK_QT_DataVisualization"
    VERSION 1.0.1
    DESCRIPTION "Process images and create 3D computer graphics with the Visualization Toolkit via Modern C++"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX C
)

# =============================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 只启用 ISO C 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(UNIX)
    message(STATUS "---->Current Unix/Linux platform")
elseif(WIN32)
    message(STATUS "---->Current Windows platform")
else()
    message(STATUS "---->other platform: ${CMAKE_SYSTEM_NAME}")
endif()

# set(CMAKE_BUILD_TYPE "Debug") # "Release" | "Debug"
if(CMAKE_BUILD_TYPE)
    message(STATUS "---->The build type is ${CMAKE_BUILD_TYPE}")
endif()

# message(STATUS "---->Is the C++ compiler loaded? ${CMAKE_CXX_COMPILER_LOADED}")
# message(STATUS "---->Is the C compiler loaded? ${CMAKE_C_COMPILER_LOADED}")
# if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#     message(STATUS "---->LLVM Clang compiler: ${CMAKE_CXX_COMPILER_ID}")
#     message(STATUS "---->C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
#     message(STATUS "---->C compiler version is: ${CMAKE_C_COMPILER_VERSION}")
#     add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
#     if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#         add_compile_options(-g -Wall -Wextra -Wpedantic -Werror)
#     elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
#         add_compile_options(-O2 -Wall -Wextra -Wpedantic -Werror)
#     else() # "MinSizeRel" and "RelWithDebInfo"
#         add_compile_options(-O2 -Wall -Wextra -Wpedantic -Werror)
#     endif()
# elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#     message(STATUS "---->GNU GCC compiler: ${CMAKE_CXX_COMPILER_ID}")
#     message(STATUS "---->C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
#     message(STATUS "---->C compiler version is: ${CMAKE_C_COMPILER_VERSION}")
#     add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
#     if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#         add_compile_options(-g -Wall -Wextra -Wpedantic -Werror)
#     elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
#         add_compile_options(-O2 -Wall -Wextra -Wpedantic -Werror)
#     else() # "MinSizeRel" and "RelWithDebInfo"
#         add_compile_options(-O2 -Wall -Wextra -Wpedantic -Werror)
#     endif()
# elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
#     message(STATUS "---->Intel C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
#     message(STATUS "---->C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
#     message(STATUS "---->C compiler version is: ${CMAKE_C_COMPILER_VERSION}")
# elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#     message(STATUS "---->Visual Studio C++ compiler(MSVC): ${CMAKE_CXX_COMPILER_ID}")
#     message(STATUS "---->C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
#     message(STATUS "---->C compiler version is: ${CMAKE_C_COMPILER_VERSION}")
#     add_compile_options(/utf-8 /MP8)
#     # add_compile_options(/source-charset:utf-8 /execution-charset:gb2312)
#     if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#         # add_compile_options(/DEBUG /W4 /permissive- /Wall /WX)
#         add_compile_options(/DEBUG /W4 /permissive- /Wall)
#     elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
#         # add_compile_options(/O2 /W4 /permissive- /Wall /WX)
#         add_compile_options(/O2 /W4 /permissive- /Wall)
#     else() # "MinSizeRel" and "RelWithDebInfo"
#         # add_compile_options(/O2 /W4 /permissive- /Wall /WX)
#         add_compile_options(/O2 /W4 /permissive- /Wall)
#     endif()
# else()
#     message(STATUS "---->Unknow the compiler: ${CXX_COMPILER}")
# endif()

# =============================================
# 查找 VTK
set(VTK_DIR "D:/DevelopTools/VTK/Install/lib/cmake/vtk-9.4/")
find_package(VTK REQUIRED)

if(NOT VTK_FOUND)
  message(FATAL_ERROR ": Unable to find the VTK build folder.")
endif()

if(NOT(TARGET VTK::GUISupportQt))
  message(FATAL_ERROR ": VTK not built with Qt support.")
endif()

if(NOT DEFINED VTK_QT_VERSION)
  set(VTK_QT_VERSION 5)
endif()

# 输出 VTK 的相关信息
message(STATUS "---->VTK version: ${VTK_VERSION}")
message(STATUS "---->VTK_INCLUDE_DIRS: ${VTK_INCLUDE_DIRS}")
message(STATUS "---->VTK_LIBRARY_DIRS: ${VTK_LIBRARY_DIRS}")
message(STATUS "---->VTK_LIBRARIES: ${VTK_LIBRARIES}")

set(qt_components Core Gui Widgets)
if(${VTK_QT_VERSION} VERSION_GREATER_EQUAL 6)
  list(APPEND qt_components OpenGLWidgets)
endif()
list(SORT qt_components)
# We have ui files, so this will also bring in the macro:
#   qt5_wrap_ui or qt_wrap_ui from Widgets.
find_package(Qt${VTK_QT_VERSION} QUIET
  REQUIRED COMPONENTS ${qt_components}
)

foreach(_qt_comp IN LISTS qt_components)
  list(APPEND qt_modules "Qt${VTK_QT_VERSION}::${_qt_comp}")
endforeach()

message (STATUS "VTK_VERSION: ${VTK_VERSION}, Qt Version: ${Qt${VTK_QT_VERSION}Widgets_VERSION}")

# --------------------------------------------------------------
set(CMAKE_PREFIX_PATH "D:/DevelopTools/QT/5.15.2/msvc2019_64/")

set(CMAKE_AUTOMOC ON) # 开启QT-moc支持, 信号与槽的反射机制
set(CMAKE_AUTOUIC ON) # 开启QT-uic支持, designer设计的UI-form
set(CMAKE_AUTORCC ON) # 开启QT-rcc支持, 图片/图标等相关资源文件

find_package(QT NAMES Qt5 REQUIRED COMPONENTS Widgets Core Gui OpenGL OpenGLWidgets)

# =============================================
add_subdirectory("00_HelloVTK")
