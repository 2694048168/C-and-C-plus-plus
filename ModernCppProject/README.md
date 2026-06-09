# Modern Cpp Project

<div align="center">
  <p><strong>A Modern C++ Project via Modern CMake across macOS, Linux, and Windows.</strong></p>
  <p>
    <img alt="C++17+" src="https://img.shields.io/badge/C%2B%2B-17+-00599C?logo=c%2B%2B&logoColor=white">
    <img alt="CMake 4.3+" src="https://img.shields.io/badge/CMake-4.3%2B-064F8C?logo=cmake&logoColor=white">
    <img alt="Generator Ninja Multi-Config" src="https://img.shields.io/badge/Generator-Ninja_Multi--Config-111111?logo=ninja&logoColor=white">
  </p>
</div>

## Quick Start

```bash
# macOS && Linux
build.sh

# Windows
build.bat
```

## Project Layout

```text
.
├── .gitignore
├── .clang-format
├── .clang-tidy
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
└── src
    ├── External
    │   ├── spdlog
    ├── └────include
    ├── └────lib
    │   ├── OpenCV
    ├── └────include
    ├── └────lib
    ├── Core
    │   ├── include
    ├── └────Core
    ├── └──────SymbolExport.hpp
    ├── └──────HelperFunction.hpp
    ├── └── src
    ├── └────HelperFunction.cpp
    ├── └── CMakeLists.txt
    ├── Logger
    │   ├── include
    ├── └────Logger
    ├── └──────Logger.hpp
    ├── └── src
    ├── └────Logger.cpp
    ├── └── CMakeLists.txt
    └── Application
    │   ├── Config
    ├── └──────Config.hpp
    ├── └──────Config.cpp
    ├── └── Application.cpp
    ├── └── Application.hpp
    ├── └── main.cpp
```
