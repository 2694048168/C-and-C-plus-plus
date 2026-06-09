# 设置可执行程序的名称(project in VS)
project(ModernCppApp)

# 子目录
add_subdirectory(Core)
add_subdirectory(Logger)

# 列出源文件-显式指定
set(SOURCES_FILES
    Application/Application.cpp
    Application/main.cpp
    Application/Config/Config.cpp
)
# 构建可执行程序
add_executable(${PROJECT_NAME}
    ${SOURCES_FILES}
)
set_target_properties(${PROJECT_NAME} 
    PROPERTIES DEBUG_POSTFIX "d"
)
# 对应编译时所需要的头文件路径
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/Application/
)
# 对应链接时所需要的库文件
target_link_libraries(${PROJECT_NAME} 
    PRIVATE 
        IthacaCore
        IthacaLogger
)
