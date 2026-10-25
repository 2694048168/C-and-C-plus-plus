# 设置可执行程序的名称(project in VS)
project(AIModernCpp)

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

# =========================================
add_subdirectory(tutorial_json)
add_subdirectory(tutorial_http)
