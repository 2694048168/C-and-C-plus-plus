# Git utility module
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

project(ex01_git_module
  VERSION 1.0
  DESCRIPTION "git utility module example"
  LANGUAGES CXX
)

# Add cmake folder to the module path, so subsequent include() calls
# can directly include modules under cmake/ folder by specifying the name only.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/)

add_subdirectory(test-executable)
