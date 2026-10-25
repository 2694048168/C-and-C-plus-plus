# 设置可执行程序的名称(project in VS)
project(tutorial_http)

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
target_compile_definitions(${PROJECT_NAME} PRIVATE 
    CPPHTTPLIB_OPENSSL_SUPPORT
)
# 对应链接时所需要的库文件
target_link_libraries(${PROJECT_NAME} PRIVATE
    nlohmann_json::nlohmann_json
    httplib::httplib
    OpenSSL::SSL 
    OpenSSL::Crypto
)

# copy OpenSLL dll files
file(GLOB OPENSSL_DLLS "D:/DevelopTools/OpenSSL-Win64/bin/*.dll")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${OPENSSL_DLLS}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying OpenSSL DLLs to output directory"
)
