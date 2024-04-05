# 设置生成库的名称
project(MathLib)

# 生成动态库
add_library(${PROJECT_NAME}
    SHARED
    Math.cpp
)

target_compile_definitions(${PROJECT_NAME}
    PRIVATE
    MYMATH_API_EXPORT
)
