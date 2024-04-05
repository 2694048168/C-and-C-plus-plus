project(test_print)

add_executable(${PROJECT_NAME} test_print.cpp)

# 对应编译时所需要的头文件路径
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${Include_Dir}
)

# 对应链接时所需要的库文件
target_link_libraries(${PROJECT_NAME}
    PUBLIC
    PrintModuleLib
)

# 增加 ctest 单元测试
add_test(
    NAME ${PROJECT_NAME}
    COMMAND ${PROJECT_NAME}
)