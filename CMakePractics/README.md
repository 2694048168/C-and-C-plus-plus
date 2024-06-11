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

cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE:STRING=Release -S . -B ./build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE:STRING=Debug -S . -B ./build
cmake -G "Ninja" -DCMAKE_CXX_FLAGS:STRING="-Wall -Werror" -S . -B ./build

# see a list of available presets
run cmake --list-presets
# To build using a preset, run 
cmake --build --preset_name

# Listing cached variables
cmake -L ./build
# show the advanced variables and help strings associated with each variable
cmake -LAH ./build

# Building in parallel job_count
cmake --build ./build --parallel 8
cmake --build ./build/ --parallel $(($(nproc)-1))

# Building specific target(s)
cmake --build ./build/ --target "_framework_component1" --target "_framework_component2"

# Removing previous build artifacts before the build
cmake --build build --clean-first

# verbose mode to investigate nasty compilation and linkage errors with ease
cmake --build build --clean-first --verbose
cmake --build build --clean-first --verbose --

# additional **--prefix** parameter to change the installation directory
cmake --install build --prefix D:/DevelopTools
cmake --install build --prefix /home/weili/DevelopTools
cmake --install build --strip
cmake --install build --component component_name

# for multiple-configuration generators only, such as Visual Studio
cmake --install build --config Debug
cmake --install build --config Release

# install target into specific path, 
# and find_package via cmake folder config files
cmake --install .\build\ --prefix ..\consume_cmake\external

# Creating an installable package with CPack
cpack --config build/CPackConfig.cmake -B build/


```

### Reference
- [**CMake Best Practices**](https://github.com/2694048168/C-and-C-plus-plus/tree/master/CMakePractics)
- [CMake Download](https://cmake.org/download/)
- [Ninja Download](https://ninja-build.org/)
- [Git Download](https://www.git-scm.com/downloads)
- [VSCode Download](https://code.visualstudio.com/Download)
- [CMake Tool extension for VSCode](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)


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
- [CMake Properties](https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html)
- [CMake Generate Expressions](https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html) think of generator expressions as small inline if-statements
- [CMake presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html), **CMakePresets.json** files, which are placed in the root directory of a project
- each user can superimpose their configuration with a **CMakeUserPresets.json** file

### 01_Accessing_CMake_Best_Ways
- Configuring, Building, and Installing the CMake projects
- to interact with CMake projects via **command-line interface (CLI)**
- cmake version <maj.min.rev>
- cmake -G "Unix Makefiles" -S <project_root> -B <output_directory>
- To supply additional variables, the variable must be prefixed with -D
- cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE:STRING=Release -S . -B ./build

> NOTE: The CMAKE_BUILD_TYPE variable only makes sense for single-configuration generators, such as Unix Makefiles and Ninja. In multiple-configuration generators, such as Visual Studio, the build type is a build-time parameter instead of a configuration-time parameter, thus, it cannot be configured by using the CMAKE_BUILD_TYPE parameter. 

- cmake -G "Ninja" -DCMAKE_BUILD_TYPE:STRING=Debug -S . -B ./build
- compilers used per-language via the **CMAKE_<LANG>_COMPILER** variables
- CMAKE_C_COMPILER || CMAKE_CXX_COMPILER || CMAKE_CUDA_COMPILER
- Compiler flags are similarly controlled by the **CMAKE_<LANG>_FLAGS** variable
- GCC or Clang enable all warnings and treat them as an error with -Wall and -Werror compiler flags
- cmake -G "Ninja" -DCMAKE_CXX_FLAGS:STRING="-Wall -Werror" -S . -B ./build
- Build flags can be customized for a per-build type by suffixing them with capitalized build type string
- **CMAKE_<LANG>_FLAGS_DEBUG** | **CMAKE_<LANG>_FLAGS_RELEASE** | CMAKE_<LANG>_FLAGS_RELWITHDEBINFO | CMAKE_<LANG>_FLAGS_MINSIZEREL
- cmake -G "Unix Makefiles" -DCMAKE_CXX_FLAGS:STRING="-Wall -Werror" -DCMAKE_CXX_FLAGS_RELEASE:STRING="-O3" -DCMAKE_BUILD_TYPE:STRING= "Release" -S . -B ./build
- The benefit of using **cmake --build** via invoking build system-specific commands helpful when building CI pipelines or build scripts
- Building in parallel append **--parallel <job_count>** to cmake --build command

> one job per hardware thread, In multi-core systems, it is also recommended to use at least one less than the available hardware thread count to not affect the system's responsivity during the build process. You can usually use more than one job per hardware thread and get faster build times since the build process is mostly I/O bound, but your mileage may vary. 

- cmake --build ./build/ --parallel $(($(nproc)-1))
- Building specific target(s) only via **--target** sub-option with multiple times
- Removing previous build artifacts before the build via **--clean-first** sub-option
- Debugging your build process via **--verbose** sub-command instructs CMake to invoke all build commands with verbose mode to investigate nasty compilation and linkage errors with ease

- Passing command-line arguments to the build tool via **--**, such **--trace** to make build tool
- CMake code must be already using CMake **install()** instructions to specify what to install when **cmake --install** (or the build system equivalent) is invoked
- default installation directory varies between environments, For Unix-like environments, it defaults to **/usr/local**, whereas in a Windows environment, it defaults to **C:/Program Files**
- additional **--prefix** parameter to change the installation directory
- cmake --install build --prefix /tmp/example
- CMake's --install command allows the stripping of binaries while installing the operation via **--strip**
- Installing specific components only (component-based install) via **--component** argument
- Installing a specific configuration (for multiple-configuration generators only), such as Visual Studio
- cmake --install build --config Debug/Release 
- Our motto here is, Being **explicit** is almost always better than being **implicit** 
- ccmake (CMake curses GUI) and cmake-gui is a terminal-based graphical user interface (GUI) for CMake
- ccmake -G "Unix Makefiles" -S . -B ./build
- Using CMake in Visual Studio, Visual Studio Code, and Qt Creator
- Passing arguments to the debugged target[VSCode's settings.json]
- always define additional kits by adding them to the user-local **cmake-tools-kits.json** file manually

### 02_Creating_CMake_Project
- **build**: The folder where the build files and binaries are placed
- **include/project_name**: This folder contains all the header files that are publicly accessible from outside the project. Adding a subfolder that contains the project's name is helpful since includes are done with <project_name/some_file.h>, making it easier to figure out which library a header file is coming from.
- **src**: This folder contains all the source and header files that are private
- **CMakeLists.txt**: This is the root CMake file

```
./project_name
├── CMakeLists.txt
├── build
├── docs
├── test
├── external or 3thParty
├── include/project_name
└── src
└── sub_project
    ├── CMakeLists.txt
    ├── include
    │   └── sub_project
    └── src
```

- [the available compile features](https://cmake.org/cmake/help/latest/prop_gbl/CMAKE_CXX_KNOWN_FEATURES.html)
- **PUBLIC**, **PRIVATE**, **INTERFACE** for source and header files

> Naming libraries, using add_library(<name>), the platform, such as lib<name>.so on Linux and <name>.lib or <name>.dll on Windows.  the prefix or postfix of lib as CMake may append or prepend the appropriate string to the filename, depending on the platform.

- Symbol(classes, functions, types, and more) visibility in shared libraries
- Compilers have different ways and default behavior when specifying symbol visibility

> the default visibility of the compilers; gcc and clang assume that all the symbols are visible, while Visual Studio compilers, by default, hide all the symbols unless they're explicitly exported. By setting "CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS", the default behavior of MSVC can be changed.

- Changing the default visibility via "<LANG>_VISIBILITY_PRESET" property to HIDDEN
- CMake provides **generate_export_header** macro, which is imported by the **GenerateExportHeader** module
- sets the "VISIBILITY_INLINES_HIDDEN" property to TRUE to further reduce the export symbol table by hiding  inlined class member functions

> It is good practice to put these generated files in a subfolder of the build directory so that only part of the directory is added to the include path. The include structure of the generated files should match the include structure of the rest of the library.

- [Additional information about setting symbol visibility via CMake](https://cmake.org/cmake/help/latest/module/GenerateExportHeader.html)
- Interface or header-only libraries via  **add_library(MyLibrary INTERFACE)**
- Object libraries – for internal use only via  **add_library(MyLibrary OBJECT)**
- Setting compiler and linker options via **target_compile_options** and **target_link_options**

> GCC and Clang, options are passed with a dash (-), while the Microsoft compiler takes slashes (/) as prefixes for its options. But by using generator expressions. Generator expressions are evaluated during build system generation, the inner expression evaluates to true and someOption is passed to the compiler. Passing compiler or linker options as PRIVATE marks them as a build requirement for this target that is not needed for interfacing the library. if PRIVATE is substituted with PUBLIC, then the compile option also becomes a usage  requirement and all the targets that depend on the original targets will use the  same compiler options. Exposing compiler options to the dependent targets is  something that needs to be done with caution. If a compiler option is only needed to use a target but not to build it, then keyword INTERFACE can be used. This is mostly the case when you're building header-only libraries. 

```CMake
target_compile_options(
  target_name
  PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/SomeOption>
          $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-someOption>
)
```

- A special case of compiler options is compile definitions, which are passed to the underlying program. These are passed with the **target_compile_definitions** function
- Debugging compiler options via **CMAKE_EXPORT_COMPILE_COMMANDS** variable to generate [**compile_commands.json**](https://clang.llvm.org/docs/JSONCompilationDatabase.html) for VSCode or CLion
- Library aliases(MyProject::Library) for common targets libraries named utils, helpers, and similar

```CMake
add_library(Namespace::utils ALIAS utils)
add_library(Namespace::utils ALIAS helpers)

target_link_libraries(SomeLibrary PRIVATE Namespace::utils)
target_link_libraries(SomeLibrary PRIVATE Namespace::helpers)
```

> As a good practice, always ALIAS your targets with a namespace and reference them using the namespace:: prefix.

### 03_Packaging_Deploying_Installing
- Building a software project is only half the story. The other half is about delivering and presenting the software to your consumers
- Remember, happy consumers will bring value to a product
- Making CMake targets installable via **install()** command that allows you to generate build system instructions for installing targets, files, directories, and more
- The **TARGETS** parameter denotes that install will accept a set of CMake targets to generate the installation code for.
- **signature**: install(TARGETS <target>... [...])
- The most common output artifacts for a target:
  - ARCHIVE (static libraries, DLL import libraries, and linker import files):
    - Except for targets marked as FRAMEWORK in macOS
  - LIBRARY (shared libraries):
    - Except for targets marked as FRAMEWORK in macOS
    - Except for DLLs (in Windows)  
  - RUNTIME (executables and DLLs):
    - Except for targets marked as MACOSX_BUNDLE in macOS
- The **GNUInstallDirs** module defines various **CMAKE_INSTALL_** paths when included default installation directories for the targets:

![](images/default_install.png)

- To override the built-in defaults, an additional **<TARGET_TYPE> DESTINATION** parameter is required in the install(...) command
- Installing files and directories via **install(FILES...)** and **install(DIRECTORY...)** commands for installing any specific files or directories(images, assets, resource files, scripts, and configuration files)

![](images/install_files.png)

> install(DIRECTORY...) the FILES_MATCHING parameter to define criteria for file selection. FILES_MATCHING can be followed by either the PATTERN or REGEX argument. PATTERN allows you to define a global pattern, whereas REGEX allows you to define a regular expression. install() command's first parameter indicates what to install. There are additional parameters that allow us to customize the installation. such "DESTINATION", "PERMISSIONS", "CONFIGURATIONS", "OPTIONAL".

- Supplying configuration information for others using your project via **find_package()** method
- Packages can be in the form of **Config-file** packages, Find-module packages, or pkg-config packages
- There are two types of configuration files: a package configuration file and an optional package version file
- Package configuration files can be named **<ProjectName>Config.cmake** or **<projectname>-config.cmake**
- Both notations will be picked by CMake on find_package(ProjectName)/find_package(projectname) calls
- **<ProjectName>ConfigVersion.cmake** or **<projectname>-config-version**
- find_package(...) looks while searching for packages is the **<CMAKE_PREFIX_PATH>/cmake** directory
- Creating an installable package with CPack to generate platform-specific installations and packages
- [available CPack generator types](https://cmake.org/cmake/help/latest/manual/cpack-generators.7.html)

| Generator Name | Description                  |
|----------------|------------------------------|
| 7Z             | 7-zip archive                |
| DEB            | Debian package               |
| External       | CPack external package       |
| IFW            | Qt Install Framework         |
| NSIS           | Null Soft Installer          |
| NSIS64         | Null Soft Installer(64-bit)  |
| NuGet          | NuGet packages               |
| RPM            | RPM packages                 |
| STGZ           | Self-extracting TAR gzip archive |
| TBZ2           | Tar BZip2 archive            |
| TGZ            | Tar GZip  archive            |
| TXZ            | Tar XZ    archive            |
| TZ             | Tar Compress archive         |
| TZST           | Tar Zstandard archive        |
| ZIP            | ZIP archive                  |

- CPack uses the configuration details that are present in the **CPackConfig.cmake** and **CPackSourceConfig.cmake** files to generate packages
- CPack module allows to customize the packaging process via a large amount of CPack variables

![](images/CPack_variables1.png)
![](images/CPack_variables2.png)

> NOTE: Any changes that must be made to the variables must be made before you include the CPack module. Otherwise, the defaults will be used.

```shell
cd CPack
cmake –S . -B build
ls build/CPack*

# for single-config generator
cmake --build build

# for multi-config generator
cmake --build build --config Release

cpack --config build/CPackConfig.cmake -B build/
```


