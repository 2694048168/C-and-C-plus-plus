cmake_minimum_required(VERSION 3.20.0)

project("ModernQTCourse"
    VERSION 1.0.1
    DESCRIPTION "QT quick course via Modern C++"
    HOMEPAGE_URL "https://github.com/2694048168/C-and-C-plus-plus"
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_C_COMPILER clang) # clang | gcc | MSVC(cl)
# set(CMAKE_CXX_COMPILER g++) # clang++ | g++ | | MSVC(cl)

# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 只启用 ISO C 标准的编译器标志, 而不使用特定编译器的扩展
# set(CMAKE_C_STANDARD 18)
# set(CMAKE_C_STANDARD_REQUIRED ON)
# set(CMAKE_C_EXTENSIONS OFF)

# 判断操作系统
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(STATUS "Linux platform: ${CMAKE_SYSTEM_NAME}")
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "Windows platform: ${CMAKE_SYSTEM_NAME}")
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    message(STATUS "Darwin platform: ${CMAKE_SYSTEM_NAME}")
else()
    message(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")
endif(CMAKE_SYSTEM_NAME MATCHES "Linux")

# add_compile_definitions(<definition> ...)
# 将预编译参数添加到源文件的编译中，对下级子目录同样有效
# 预编译命令会添加到COMPILE_DEFINITIONS目录属性中

# 判断编译器
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    message(STATUS "LLVM Clang compiler: ${CMAKE_CXX_COMPILER_ID}")

    # lots of warnings and all warnings as errors
    # add_compile_options(-Wall -Wextra -pedantic -Werror -finput-charset=utf-8 -fexec-charset=utf-8)
    add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(STATUS "GNU GCC compiler: ${CMAKE_CXX_COMPILER_ID}")

    # lots of warnings and all warnings as errors
    # add_compile_options(-Wall -Wextra -pedantic -Werror -finput-charset=utf-8 -fexec-charset=utf-8)
    add_compile_options(-finput-charset=utf-8 -fexec-charset=utf-8)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(STATUS "Intel C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    message(STATUS "Visual Studio C++ compiler(MSVC): ${CMAKE_CXX_COMPILER_ID}")

    # add_compile_options(<option> ...)
    # option:编译选项,注意对于不同编译器,支持的选项可能不一样
    # 将编译器选项添加到当前及子目录的源文件的编译中
    # warning level 4 and all warnings as errors
    # =====If you want to set both the source character set and
    # the execution character set to UTF-8, you can use the /utf-8
    # compiler option as a shortcut. It's equivalent to
    # /source-charset:utf-8 /execution-charset:utf-8 on the command line.
    # add_compile_options(/W4 /WX /source-charset:utf-8 /execution-charset:utf-8)
    # add_compile_options(/W4 /WX /utf-8)
    add_compile_options(/utf-8)
endif()

# =========================================
# 设置系统的环境变量,或者直接设置CMake变量
# set(CMAKE_PREFIX_PATH  "/home/user/Qt6/6.4.2/gcc_64/lib/cmake")
# set(CMAKE_PREFIX_PATH "D:/DevelopTools/QT/6.2.4/mingw_64/lib/cmake")
# set(CMAKE_PREFIX_PATH "D:/DevelopTools/QT/6.2.4/msvc2019_64/lib/cmake")
# set(CMAKE_PREFIX_PATH "D:/DevelopTools/QT/6.2.4/msvc2019_64/")
set(CMAKE_PREFIX_PATH "D:/DevelopTools/QT/6.2.4/mingw_64/")

set(CMAKE_AUTOMOC ON) # 开启QT-moc支持, 信号与槽的反射机制
set(CMAKE_AUTOUIC ON) # 开启QT-uic支持, designer设计的UI-form
set(CMAKE_AUTORCC ON) # 开启QT-rcc支持, 图片/图标等相关资源文件

# QT6开始直接支持CMake, 根据QT6的文档直接抄过来, 保证可以检索都头文件路径即可
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Network Sql OpenGLWidgets OpenGL)

# =========================================
# QT6开始直接支持CMake, 根据QT6的文档直接抄过来, 保证可以链接到库
add_executable("00_dev_env" 00_dev_env.cpp src/mainwindow.cpp src/mainwindow.ui)
target_link_libraries("00_dev_env" PRIVATE Qt6::Widgets)

add_executable("01_data_type" 01_data_type.cpp)
target_link_libraries("01_data_type" PRIVATE Qt6::Widgets)

set_property(SOURCE 02_signal_slot.cpp PROPERTY SKIP_AUTOMOC ON)
add_executable("02_signal_slot" 02_signal_slot.cpp)
target_link_libraries("02_signal_slot" PRIVATE Qt6::Widgets Qt6::Core)

set_property(SOURCE 03_timer.cpp PROPERTY SKIP_AUTOMOC ON)
add_executable("03_timer" 03_timer.cpp)
target_link_libraries("03_timer" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("04_fund_widget" 04_fund_widget.cpp)
target_link_libraries("04_fund_widget" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("05_layout_widget" 05_layout_widget.cpp)
target_link_libraries("05_layout_widget" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("06_right_click" 06_right_click.cpp)
target_link_libraries("06_right_click" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("07_widget_btn" 07_widget_btn.cpp)
target_link_libraries("07_widget_btn" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("08_widget_container" 08_widget_container.cpp)
target_link_libraries("08_widget_container" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("09_event_processor" 09_event_processor.cpp src/mainwindow_event.cpp src/mainwindow_event.ui)
target_link_libraries("09_event_processor" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("10_event_distributor" 10_event_distributor.cpp)
target_link_libraries("10_event_distributor" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("11_event_filter" 11_event_filter.cpp)
target_link_libraries("11_event_filter" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("12_socket_TCP" 12_socket_TCP.cpp src/TCP_client.cpp src/TCP_server.cpp src/TCP_client.ui src/TCP_server.ui)
target_link_libraries("12_socket_TCP" PRIVATE Qt6::Widgets Qt6::Core Qt6::Network)

add_executable("13_thread_task" 13_thread_task.cpp src/custom_qthread.cpp src/Thread_test.cpp src/Thread_test.ui src/Task_test.cpp src/Task_test.ui src/custom_task.cpp)
target_link_libraries("13_thread_task" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("14_thread_pool" 14_thread_pool.cpp)
target_link_libraries("14_thread_pool" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("15_sql_test" 15_sql_test.cpp)
target_link_libraries("15_sql_test" PRIVATE Qt6::Widgets Qt6::Core Qt6::Sql)

add_executable("16_database_test" 16_database_test.cpp)
target_link_libraries("16_database_test" PRIVATE Qt6::Widgets Qt6::Core Qt6::Sql)

add_executable("17_json_test" 17_json_test.cpp)
target_link_libraries("17_json_test" PRIVATE Qt6::Widgets Qt6::Core Qt6::Sql)

add_executable("18_sine_curve_drawing" 18_sine_curve_drawing.cpp src/sine_curve.cpp src/sine_curve.ui)
target_link_libraries("18_sine_curve_drawing" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("19_login_widget" 19_login_widget.cpp src/login_widget.cpp src/login_widget.ui src/login_widget.qrc)
target_link_libraries("19_login_widget" PRIVATE Qt6::Widgets Qt6::Core)

add_executable("20_graph_render" 20_graph_render.cpp src/GraphRender.cpp src/GraphRender.ui src/GraphRenderWidget.cpp)
target_link_libraries("20_graph_render" PRIVATE Qt6::Widgets Qt6::Core Qt6::OpenGLWidgets Qt6::OpenGL)
target_include_directories("20_graph_render" PRIVATE src/)

add_executable("21_image_render" 21_image_render.cpp src/ImageOpenGLWidget.cpp src/ImageRenderWidget.cpp src/ImageRender.ui src/ImageRender.cpp)
target_link_libraries("21_image_render" PRIVATE Qt6::Widgets Qt6::Core Qt6::OpenGLWidgets Qt6::OpenGL)
target_include_directories("21_image_render" PRIVATE src/)

# language switch demo
add_executable("language_switch" LanguageSwitch/main.cpp LanguageSwitch/mainwindow.ui LanguageSwitch/mainwindow.cpp)
target_link_libraries("language_switch" PRIVATE Qt6::Widgets Qt6::Core)
target_include_directories("language_switch" PRIVATE LanguageSwitch/)
