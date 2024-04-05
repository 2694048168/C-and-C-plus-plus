# 设置生成库的名称
project(PrintModuleLib)

# 生成静态块
add_library(${PROJECT_NAME}
    STATIC
    Module.cpp
)
