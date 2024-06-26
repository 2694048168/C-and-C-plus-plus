project(LoggingExamples)

# ======== glog ========
add_executable(test_glog test_glog.cpp)

# static library for glog: -DGLOG_NO_EXPORT
# shared library for glog: -DGLOG_USE_GLOG_EXPORT
target_compile_definitions(test_glog
    PRIVATE
    -DGLOG_USE_GLOG_EXPORT
)
target_include_directories(test_glog
    PUBLIC
    ${Include_Dir}
)
target_link_directories(test_glog
    PUBLIC
    ${Link_Dir}
)

if(Build_Mode STREQUAL "Debug")
    message(STATUS "======== Debug mode")
    set(Link_Libs glogd.lib)
elseif(Build_Mode STREQUAL "Release")
    message(STATUS "======== Release mode")
    set(Link_Libs glog.lib)
endif()

message(STATUS "======== Link Librarys: ${Link_Libs}")

target_link_libraries(test_glog
    PUBLIC
    ${Link_Libs}
)

# ======== log4cplus ========
add_executable(test_log4cplus test_log4cplus.cpp)

# shared library for log4cplus: -DLOG4CPLUS_BUILD_DLL
target_compile_definitions(test_log4cplus
    PRIVATE

    # -DLOG4CPLUS_BUILD_DLL
    -DUNICODE
)
target_include_directories(test_log4cplus
    PUBLIC
    ${Include_Dir}
)
target_link_directories(test_log4cplus
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

target_link_libraries(test_log4cplus
    PUBLIC
    ${Link_Libs}
)

# ======== spdlog ========
add_executable(test_spdlog test_spdlog.cpp)

target_include_directories(test_spdlog
    PUBLIC
    ${Include_Dir}
)
target_link_directories(test_spdlog
    PUBLIC
    ${Link_Dir}
)

if(Build_Mode STREQUAL "Debug")
    message(STATUS "======== Debug mode")
    set(Link_Libs spdlogd.lib)
elseif(Build_Mode STREQUAL "Release")
    message(STATUS "======== Release mode")
    set(Link_Libs spdlog.lib)
endif()

message(STATUS "======== Link Librarys: ${Link_Libs}")

target_link_libraries(test_spdlog
    PUBLIC
    ${Link_Libs}
)

# ======== logger ========
add_executable(test_logger test_logger.cpp)

target_include_directories(test_logger
    PUBLIC
    ${Include_Dir}
)
target_link_directories(test_logger
    PUBLIC
    ${Link_Dir}
)
target_link_libraries(test_logger
    PUBLIC
    logger.lib
    log4cplusU.lib
)
