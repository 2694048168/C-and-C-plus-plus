cmake_minimum_required(VERSION 3.25)

project(ProtobufCMakeProject
    VERSION 2.0.1
    DESCRIPTION "The Protocol Buffers via C++ Project"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX
)

# MinGW+Clang | Ninja | CMake | VSCode + clangd + c/c++ Snippets
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------
# 设置一系列的变量, 顶层 CMake 的变量可以被底层 CMake 访问
# 设置输出二进制路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
# set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(Include_Dir ${CMAKE_SOURCE_DIR}/)

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
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib)

# -------------------------------------------
# 查找Protobuf包
# find_package(Protobuf REQUIRED)：确保CMake能够找到Protobuf包
# 可能需要设置Protobuf_DIR变量以指向你的Protobuf安装目录
set(Protobuf_PREFIX_PATH
    "D:/DevelopTools/Protobuf/Release/include"
    "D:/DevelopTools/Protobuf/Release/lib"
    "D:/DevelopTools/Protobuf/Release/bin")
list(APPEND CMAKE_PREFIX_PATH "${Protobuf_PREFIX_PATH}")
find_package(Protobuf REQUIRED)

# external 第三方库
if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(Portobuf_DIR "D:/DevelopTools/Protobuf/Debug/")
elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(Portobuf_DIR "D:/DevelopTools/Protobuf/Release/")
else()
    set(Portobuf_DIR "D:/DevelopTools/Protobuf/Release/")
endif()

list(APPEND CMAKE_MODULE_PATH "${Portobuf_DIR}/cmake")
message(STATUS "[==== Protobuf DIR]: ${Portobuf_DIR}")

file(GLOB PROTO_FILES
    "config_params/Address.proto"
    "config_params/Person.proto"
)
message(STATUS "PROTO_FILES==" ${PROTO_FILES})

FOREACH(proto_file ${PROTO_FILES})
    STRING(REGEX REPLACE "[^/]proto" "" proto_file_name ${proto_file})
    LIST(APPEND PROTO_SRCS "${proto_file_name}.pb.cc")
	ADD_CUSTOM_COMMAND(
		OUTPUT "${proto_file_name}.pb.h" "${proto_file_name}.pb.cc"
		COMMAND protoc --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/config_params
                       --cpp_out=${CMAKE_CURRENT_SOURCE_DIR}/config_params/ ${proto_file} 
		DEPENDS ${proto_file}
	)
ENDFOREACH(proto_file)

# --------------------------------------------------
add_executable(${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
    PUBLIC ${Include_Dir}
    PUBLIC ${Portobuf_DIR}/include
)
target_compile_definitions(${PROJECT_NAME}
    PRIVATE
    "PROTOBUF_USE_DLLS"
)
target_sources(${PROJECT_NAME}
    PRIVATE
    main.cpp
    config_params/Address.pb.cc
    config_params/Person.pb.cc
)

if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
    target_link_libraries(${PROJECT_NAME}
        PUBLIC
        ${Portobuf_DIR}/lib/libprotocd.lib
        ${Portobuf_DIR}/lib/libprotobuf-lited.lib
        ${Portobuf_DIR}/lib/libprotobufd.lib
    )
elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
    target_link_libraries(${PROJECT_NAME}
        PUBLIC
        ${Portobuf_DIR}/lib/libprotoc.lib
        ${Portobuf_DIR}/lib/libprotobuf-lite.lib
        ${Portobuf_DIR}/lib/libprotobuf.lib
    )
else()
    target_link_libraries(${PROJECT_NAME}
        PUBLIC
        ${Portobuf_DIR}/lib/libprotoc.lib
        ${Portobuf_DIR}/lib/libprotobuf-lite.lib
        ${Portobuf_DIR}/lib/libprotobuf.lib
    )
endif()

# ----------------- copy dll into bin ---------------
# 把 "D:\DevelopTools\Protobuf\bin" 指代的所有dll都拷贝到可执行文件目录下
if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
    file(GLOB BRARYS_DYNAMIC
        "${Portobuf_DIR}/bin/libprotocd.dll"
        "${Portobuf_DIR}/bin/libprotobuf-lited.dll"
        "${Portobuf_DIR}/bin/libprotobufd.dll"
    )
elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
    file(GLOB BRARYS_DYNAMIC
        "${Portobuf_DIR}/bin/libprotoc.dll"
        "${Portobuf_DIR}/bin/libprotobuf-lite.dll"
        "${Portobuf_DIR}/bin/libprotobuf.dll"
    )
else()
    file(GLOB BRARYS_DYNAMIC
        "${Portobuf_DIR}/bin/libprotoc.dll"
        "${Portobuf_DIR}/bin/libprotobuf-lite.dll"
        "${Portobuf_DIR}/bin/libprotobuf.dll"
    )
endif()

file(COPY ${BRARYS_DYNAMIC} DESTINATION "${EXECUTABLE_OUTPUT_PATH}")
