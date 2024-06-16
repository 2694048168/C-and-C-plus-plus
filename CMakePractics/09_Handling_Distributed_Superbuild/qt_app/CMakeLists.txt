# Building a QT project with Superbuild
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25.0)

project(simple_qt_app
  VERSION 1.0
  DESCRIPTION "Building a QT Project With Superbuild"
  LANGUAGES CXX
)

# Adapted from the following example: https://doc.qt.io/qt-5/cmake-get-started.html#build-a-gui-executable
if(USE_SUPERBUILD)
  include(superbuild.cmake)
else()
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTORCC ON)
  set(CMAKE_AUTOUIC ON)

  # User decided to not to use superbuild. We have to discover the dependencies
  # from the current environment as usual.
  # We need `Core` and Widgets QT components for our simple QT app.
  find_package(Qt6 COMPONENTS Core Widgets REQUIRED)
endif()

# Define the executable target
add_executable(simple_qt_app
  main.cpp
)
target_compile_features(simple_qt_app
  PRIVATE cxx_std_17
)

# Link QT components to the executable
target_link_libraries(simple_qt_app
  Qt6::Core Qt6::Widgets
)
