cmake_minimum_required(VERSION 3.25)

project(spdlog_encapsulation
    LANGUAGES CXX
)

add_executable(${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/
)
target_sources(${PROJECT_NAME}
    PRIVATE
    main_test.cpp
    logger.cpp
)
