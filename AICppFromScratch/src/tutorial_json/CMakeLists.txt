# 设置可执行程序的名称(project in VS)
project(tutorial_json)

# 列出源文件-显式指定
set(SOURCES_FILES
    main.cpp
)
# 构建可执行程序
add_executable(${PROJECT_NAME}
    ${SOURCES_FILES}
)
set_target_properties(${PROJECT_NAME} 
    PROPERTIES DEBUG_POSTFIX "d"
)
# 对应链接时所需要的库文件
target_link_libraries(${PROJECT_NAME} PRIVATE
    nlohmann_json::nlohmann_json
)
