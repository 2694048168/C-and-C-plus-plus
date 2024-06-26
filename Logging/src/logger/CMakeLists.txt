project(logger)

# 生成动态库
add_library(${PROJECT_NAME}
    STATIC
    logger.cpp
)

# shared library for log4cplus: -DLOG4CPLUS_BUILD_DLL
target_compile_definitions(${PROJECT_NAME}
    PRIVATE
    -DLOG4CPLUS_BUILD_DLL
    -DUNICODE
)
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${Include_Dir}
)
target_link_directories(${PROJECT_NAME}
    PUBLIC
    ${Link_Dir}
)

if(Build_Mode STREQUAL "Debug")
    message(STATUS "======== Debug mode")
    set(Link_Libs log4cplusUD.lib)
elseif(Build_Mode STREQUAL "Release")
    message(STATUS "======== Release mode")
    set(Link_Libs log4cplusU.lib)
endif()

message(STATUS "======== Link Librarys: ${Link_Libs}")

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${Link_Libs}
)
