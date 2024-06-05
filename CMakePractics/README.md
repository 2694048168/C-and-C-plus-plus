## CMake Best Practices

> Discover proven techniques for creating and maintaining programming projects with CMake

### CMake Command
```shell
cmake --version
cmake --help
cmake -S . -B build
cmake --build build
cmake -S . -B build -G Ninja
cmake -G

```

### Reference
- [**CMake Best Practices**](https://github.com/2694048168/C-and-C-plus-plus/tree/master/CMakePractics)
- [CMake Download](https://cmake.org/download/)
- [Ninja Download](https://ninja-build.org/)
- [Git Download](https://www.git-scm.com/downloads)


### Features
- **an industry standard** to building C++ applications
- **software engineers first** and more practical than theoretical
- install **latest stable** CMake and system **generator**
- **command line** and GUI and integrates with IDEs
- CMake project with **executable and library** and link together
- Packaging Deploying and Installing with **CPack(CMake's packaging program)**
- integrating **Third-Party libraries and Dependency** management
- Automatically generating Documentation with doxygen, dot (graphviz) and plantuml
- Integrating Code-Quality Tools(**unit testing**, code sanitizers, static code analysis, and code coverage tools)
- Custom Tasks, configuration the platform-agnostic commands
- Reproducible Build Environments(**CI/CD pipelines**)
- Big Projects and Distributed Repositories(**git**)
- Cross-Platform Compiling and Custom **Toolchains**
- CMake modules and generalize **CMake files**

### 00_Kickstarting_CMake
- an industry standard and build system generator
- support build systems such as Makefile, Ninja, Visual Studio, Qt Creator, Android Studio and Xcode.
- support installing, packaging, and testing software
- CMake language to configure build processes
- CMake consists command-line tools: cmake + ctest + cpack, and interactive tools: cmake-gui + ccmake
- CMake integrates package managers such as Conan and vcpkg
![CMake-build-process](images/build_process.png)
- **single-configuration** and multi-configuration generators
- **CMakeCache.txt** file where all the detected configurations are stored
- Each folder containing a **CMakeLists.txt** file will be mapped and a subfolder called **CMakeFiles** will be created, which contains the information that's used by CMake for building
- overview of the **core concepts** and **[language features](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html)**
- Cmake language is simple and supports **variables**, string functions, **macros**, function definitions, and **importing other CMake files**
- Cmake syntax is based on **keywords** and **whitespace-separated** arguments
- The **PUBLIC** and **PRIVATE** keywords denote the visibility of the files when they're linked against this library and serve as delimiters between the lists of files
- CMake language supports **"generator expressions"**, which are evaluated during build system generation
- [CMake predefined variables prefixed with CMAKE_](https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html)
