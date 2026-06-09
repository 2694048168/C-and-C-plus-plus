# 设置生成库的名称
project(IthacaUserRole)

# 生成动态库
add_library(${PROJECT_NAME}
    SHARED
    src/DatabaseManager.cpp
)
# 符号导出
target_compile_definitions(${PROJECT_NAME}
    PRIVATE
    LIB_API_EXPORT
)
# 设置头文件搜索路径(PUBLIC 使得依赖此库的目标也能自动获得该路径)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}
    PUBLIC IthacaCore
    PUBLIC IthacaLogger
)
