set(LIB_NAME "ProjectMath")
project(${LIB_NAME})
add_library(${LIB_NAME} SHARED src/math.cpp)
target_compile_definitions(${LIB_NAME}
    PRIVATE
        "API_EXPORT"
)
