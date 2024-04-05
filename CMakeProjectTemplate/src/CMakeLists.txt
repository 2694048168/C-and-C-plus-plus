# 设置可执行程序的名称(project in VS)
project(Project_Main)

# 添加 CMake 子目录, 寻找该目录下的 CMakeLists.txt
add_subdirectory(MathLib)
add_subdirectory(PrintModuleLib)

# 构建可执行程序
add_executable(${PROJECT_NAME}
    main.cpp
)

# 对应编译时所需要的头文件路径
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${Include_Dir}
)

# # 条件判断
# if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
# file(GLOB Link_Libs
# ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG}/MathLib.lib
# ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG}/PrintModuleLib.lib
# )
# elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
# file(GLOB Link_Libs
# ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}/MathLib.lib
# ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}/PrintModuleLib.lib
# )
# else()
# file(GLOB Link_Libs
# ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}/MathLib.lib
# ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}/PrintModuleLib.lib
# )
# message(STATUS "The Linker Libs: ${Link_Libs}")
# endif()
# message(STATUS "============${Link_Libs}")

# 条件判断
if(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(Link_Lib_Dir ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG})
elseif(CMAKE_BUILD_TYPE AND(CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(Link_Lib_Dir ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE})
else()
    set(Link_Lib_Dir ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE})
endif()

message(STATUS "============${Link_Lib_Dir}")
message(STATUS "============${CMAKE_BUILD_TYPE}")

# 对应链接时所需要的库文件路径
target_link_directories(${PROJECT_NAME}
    PUBLIC
    ${Link_Lib_Dir}
)

# 对应链接时所需要的库文件
target_link_libraries(${PROJECT_NAME}
    PUBLIC
    MathLib
    PrintModuleLib
    Eigen3::Eigen
)
