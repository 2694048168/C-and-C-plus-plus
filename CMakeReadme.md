# CMake CookBook

> [CMake CookBook](https://www.bookstack.cn/read/CMake-Cookbook/README.md)

## Chapter 00 Configuration of Development Environment

**the configuration of devolopment environment**
```shell
# Modern code editor: Visual Studio Code
winget show code

# Build tool: CMake
cmake --version

# Compile tool: g++ from MinGW-64
g++ --version
mingw32-make --version

# Terminal on Windows
git bash integration into VSCode
Powershell on Windows Terminal
```

**the organization of project**
```
. CMakeExamples
|—— 01_chapter
|   |—— src
|   |—— |—— hello_cmake.cpp
|   |—— CMakeLists.txt
|—— ...
|—— 07_chapter
|   |—— 01_recipe
|   |—— |—— src
|   |—— |—— |—— main.cpp
|   |—— |—— |—— CMakeLists.txt
|   |—— |—— test
|   |—— |—— |—— test.cpp
|   |—— |—— |—— CMakeLists.txt
|   |—— CMakeLists.txt
|—— ...
|—— 14_chapter
|—— README.md
```

**Table of Contents**
- [Chapter 01 从可执行程序到库](#Chapter-01-从可执行程序到库)
- [Chapter 02 检测环境](#Chapter-02-检测环境)
- [Chapter 03 检测外部库和程序](#Chapter-03-检测外部库和程序)
- [Chapter 04 创建和运行测试](#Chapter-04-创建和运行测试)
- [Chapter 05 配置时和构建时的操作](#Chapter-05-配置时和构建时的操作)
- [Chapter 06 生成源码](#Chapter-06-生成源码)
- [Chapter 07 构建项目](#Chapter-07-构建项目)
- [Chapter 08 超级构建模式](#Chapter-08-超级构建模式)
- [Chapter 09 语言混合项目](#Chapter-09-语言混合项目)
- [Chapter 10 编写安装程序](#Chapter-10-编写安装程序)
- [Chapter 11 打包项目](#Chapter-11-打包项目)
- [Chapter 12 构建文档](#Chapter-12-构建文档)
- [Chapter 13 选择生成器和交叉编译](#Chapter-13-选择生成器和交叉编译)
- [Chapter 14 测试面板](#Chapter-14-测试面板)
- [Chapter 15 使用 CMake 构建已有项目](#Chapter-15-使用-CMake-构建已有项目)


## Chapter 01 从可执行程序到库

> CMake 构建代码所需的基本任务: 编译可执行文件、编译库、根据用户输入执行构建操作等; CMake 是一个构建系统生成器, 特别适合于独立平台和编译器; 除非另有说明, 否则所有配置都独立于操作系统, 可以在 GNU/Linux、macOS 和Windows 的系统下运行.

```shell
# =============== 操作命令行 ===============
cd 00_chapter
mkdir build # configure project in build folder.
cd build
cmake .. # building project(compile and linking).
cmake --build .

# 或者使用现代 CMake 标准一条命令(VSCode CMake Tools 插件实现的方式):
# -H. 表示当前目录中搜索根 CMakeLists.txt 文件;
# -Bbuild 告诉 CMake 在一个名为 build 的目录中生成所有的文件.
cd 00_chapter
pwd
cmake -H. -Bbuild
cmake --build build

# CMake 生成的目标比构建可执行文件的目标要多
# <target-name> == all | clean | rebuild_cache | edit_cache | test | install | package
cmake --build build --target <target-name>

# 查看 CMake 所支持的所有生成器 Generator
cmake -G
# 可以随时切换生成器,每一个生成器都有自己的文件集合，故此编译步骤的输出和构建目录的内容是不同的
rm build -r
cmake -G Ninja  -H. -Bbuild
cmake -G "MinGW Makefiles"  -H. -Bbuild
cmake --build build

# if()-else()-endif()
# 若逻辑变量设置为以下任意一种：1、ON、YES、true、Y或非零数，则逻辑变量为true。
# 若逻辑变量设置为以下任意一种：0、OFF、NO、false、N、IGNORE、NOTFOUND、空字符串，或者以-NOTFOUND为后缀，则逻辑变量为false。

# 通过 CMake 的 "-D" CLI选项, 控制编译链接的构建过程
# -D 开关用于为 CMake 设置任何类型的变量: 逻辑变量、路径等等
cmake -D USE_LIBRARY=ON -H. -Bbuild
cmake --build build

# CMake 将语言的编译器存储在 CMAKE_<LANG>_COMPILER 变量中,
# 其中 <LANG> 是受支持的任何一种语言: CXX、C或Fortran
cmake -D CMAKE_CXX_COMPILE=g++ -H. -Bbuild
# CMake 将在标准路径中执行查找编译器, 否则需要提供完整的编译器执行文件的路径

# CMake提供--system-information标志，它将把关于系统的所有信息转储到屏幕或文件中
cmake --system-information
cmake --system-information information.txt

# CMake 配置构建类型 'CMAKE_BUILD_TYPE', 默认值为空:
# 1. Debug：用于在没有优化的情况下，使用带有调试符号构建库或可执行文件
# 2. Release：用于构建的优化的库或可执行文件，不包含调试符号
# 3. RelWithDebInfo：用于构建较少的优化库或可执行文件，包含调试符号
# 4. MinSizeRel：用于不增加目标代码大小的优化方式，来构建库或可执行文件
cmake -H. -Bbuild
cmake -H. -Bbuild -D CMAKE_BUILD_TYPE=Debug
cmake -H. -Bbuild -G "MinGW Makefiles"
cmake -H. -Bbuild -D CMAKE_BUILD_TYPE=Debug -G "MinGW Makefiles"

# 对于单配置生成器，如 Unix Makefile、MSYS Makefile或Ninja,
# 因为要对项目重新配置, 这里需要运行 CMake 两次;
# CMake 也支持复合配置生成器, 这些通常是集成开发环境提供的项目文件，最显著的是Visual Studio和Xcode
mkdir -p build
cd build
cmake .. -G "Visual Studio 17 2022" -D CMAKE_CONFIGURATION_TYPES="Release;Debug"
# 将为 Release 和 Debug 配置生成一个构建树,
# 然后, 可以使 --config 标志来决定构建这两个中的哪一个:
cmake --build . --config Release
cmake --build . --config Debug

# CMake 设置编译选项 compiler flags
# 1. CMake 将编译选项视为目标属性,可以根据每个目标设置编译选项,而不需要覆盖 CMake 默认值
# 2. 使用 -D 的 Command-line interface(CLI)标志直接修改 CMAKE_<LANG>_FLAGS_<CONFIG> 变量,
#    这将影响项目中的所有目标, 并覆盖或扩展 CMake 默认值
# 编译选项可以添加三个级别的可见性：INTERFACE、PUBLIC和PRIVATE。
# 1. PRIVATE,编译选项会应用于给定的目标,不会传递给与目标相关的目标;
# 2. INTERFACE,给定的编译选项将只应用于指定目标,并传递给与目标相关的目标;
# 3. PUBLIC,编译选项将应用于指定目标和使用它的目标.
```

## Chapter 02 检测环境

> 尽管 CMake 跨平台, 但有时源代码并不是完全可移植(例如:当使用依赖于供应商的扩展时), 努力使源代码能够跨平台、操作系统和编译器; 这个过程中会发现, 有必要根据平台不同的方式配置和/或构建代码; 这对于历史代码或交叉编译尤其重要.

```shell
# step 1. 检测操作系统
# step 2. 处理与平台相关的源码
# step 3. 处理与编译器相关的源码
# step 4. 检测处理器体系结构
# step 5. 检测处理器指令集
# step 6. Eigen library 使能向量化

# 大多数处理器提供向量指令集,代码可以利用这些特性,获得更高的性能;
# 由于线性代数运算可以从 Eigen 库中获得很好的加速, 所以在使用Eigen库时,就要考虑向量化;
# 我们所要做的就是,指示编译器为我们检查处理器,并为当前体系结构生成本机指令;
# 不同的编译器供应商会使用不同的标志来实现这一点: GNU 编译器使用 '-march=native' 标志来实现这一点;
# 而 Intel 编译器使用 '-xHost' 标志;
# CMake 使用 CheckCXXCompilerFlag.cmake 模块提供的 
# "check_cxx_compiler_flag" 函数进行编译器标志的检查
check_cxx_compiler_flag("-march=native" _march_native_works)
#Params 1,要检查的编译器标志;
#Params 2,是用来存储检查结果(true或false)的变量
# 如果检查为真, 将工作标志添加到 _CXX_FLAGS 变量中, 该变量将用于为可执行目标设置编译器标志.
```

## Chapter 03 检测外部库和程序

> 项目常常会依赖于其他项目和库,如何检测外部库、框架和项目,以及如何链接到这些库; CMake 有一组预打包模块,用于检测常用库和程序, 可以使用 cmake --help-module-list 获得现有模块的列表;

```CMake
find_file：在相应路径下查找命名文件
find_library：查找一个库文件
find_package：从外部项目查找和加载设置
find_path：查找包含指定文件的目录
find_program：找到一个可执行程序
# NOTE:可以使用 --help-command 命令行显示 CMake 内置命令的打印文档

# 用户可以使用CLI的 -D 参数传递相应的选项,
# 告诉 CMake 查看特定的位置, Python解释器可以使用以下配置:
cmake -D PYTHON_EXECUTABLE=/custom/location/python ...
```

- 检测 Python 解释器、Python库、Python模块和包
> 将 Python 解释器嵌入到 C/C++ 程序中,都需要下列条件:1)Python 解释器的工作版本;2)Python 头文件 Python.h 的可用性; 3)Python 运行时库 libpython; 三个组件所使用的 Python 版本必须相同.

- 检测 BLAS 和 LAPACK 数学库
> 许多数据算法严重依赖于矩阵和向量运算,矩阵-向量和矩阵-矩阵乘法,求线性方程组的解,特征值和特征向量的计算或奇异值分解; 这些操作在代码库中非常普遍,因为操作的数据量比较大,因此高效的实现有绝对的必要; 基本线性代数子程序(BLAS)和线性代数包(LAPACK),为许多线性代数操作提供了标准API,供应商有不同的实现,但都共享API; 用于数学库底层实现,实际所用的编程语言会随着时间而变化(Fortran、C、Assembly),但是也都是 Fortran 调用接口,要链接到这些库,并展示如何用不同语言编写的库.

- 检测 OpenMP 并行环境、检测 MPI 并行环境
> 现代计算机都是多核机器,对于性能敏感的程序,必须关注这些多核处理器,并在编程模型中使用并发; OpenMP 是多核处理器上并行性的标准之一,从 OpenMP 并行化中获得性能收益,通常不需要修改或重写现有程序;一旦确定了代码中的性能关键部分,例如使用分析工具,就可以通过预处理器指令,指示编译器为这些区域生成可并行的代码(前提是使用一个支持OpenMP的编译器); 消息传递接口(Message Passing Interface,MPI),可以作为 OpenMP (共享内存并行方式)的补充,它也是分布式系统上并行程序的实际标准;尽管最新的MPI实现也允许共享内存并行,但高性能计算中的一种典型方法就是在计算节点上OpenMP与MPI结合使用.

- 检测 Eigen 库、检测 Boost 库
> 纯头文件实现的Eigen库,使用模板编程来提供接口;矩阵和向量的计算,会在编译时进行数据类型检查,以确保兼容所有维度的矩阵; 密集和稀疏矩阵的运算, 也可使用表达式模板高效的进行实现,如矩阵-矩阵乘积,线性系统求解器和特征值问题; 从3.3版开始, Eigen可以链接到BLAS和LAPACK库中,这可以将某些操作实现进行卸载,使库的实现更加灵活,从而获得更多的性能收益.

> Boost是一组C++通用库,这些库提供了许多功能,这些功能在现代C++项目中不可或缺,但是还不能通过C++标准使用这些功能;例如Boost为元编程、处理可选参数和文件系统操作等提供了相应的组件;这些库中有许多特性后来被C++11、C++14和C++17标准所采用,但是对于保持与旧编译器兼容性的代码库来说,许多Boost组件仍然是首选.

- 检测外部库:Ⅰ. 使用 pkg-config and 编写 find 模块
> 使用CMake自带的find-module,但并不是所有的包在CMake的find模块都找得到; 使用<package>Config.cmake, <package>ConfigVersion.cmake和<package>Targets.cmake, 这些文件由软件包供应商提供,并与软件包一起安装在标准位置的cmake文件夹下; 如果某个依赖项既不提供查找模块,也不提供供应商打包的CMake文件,在这种情况下,只有两个选择: 1)依赖pkg-config程序,来找到系统上的包,这依赖于包供应商在.pc配置文件中,其中有关于发行包的元数据; 2)为依赖项编写find-package模块.

## Chapter 04 创建和运行测试

> 测试代码是开发工具的核心组件,通过单元测试和集成测试自动化测试,不仅可以帮助开发人员尽早回归功能检测,还可以帮助开发人员参与,并了解项目;它可以帮助新开发人员向项目代码提交修改,并确保预期的功能性;对于验证安装是否保留了代码的功能时,自动化测试必不可少;从一开始对单元、模块或库进行测试,可以使用一种纯函数式的风格,将全局变量和全局状态最小化,可让开发者的具有更模块化、更简单的编程风格.

- CTest 是 CMake 的测试工具
- Catch2 是一个提供了相关基础以便运行复杂的测试框架
- Google Test 测试框架
- Boost Test 是 C++ 社区中一个非常流行的单元测试框架
- Valgrind 动态分析来检测内存缺陷
- CMake 如何定义预期测试失败
- 使用超时测试运行时间过长的测试
- CTest 并行测试和定制测试子集 & ctest --help 有大量的选项可供用来定制测试

## Chapter 05 配置时和构建时的操作

> CMake time(config); CMake Generation time; CMake Build time; CTest time; CDash time; CMake Install time; Package Install time.

![CMake workflow](./cmake_workflow.png)

- 使用平台无关的文件操作
- 配置时运行自定义命令: execute_process; add_custom_command; add_custom_target
- CMake 探究编译和链接命令

```shell
# TIPS:CMake 命令行界面来获取关于特定模块的文档
cmake --help-module CheckCXXSourceCompiles
```

- CMake 探究编译器标志命令 & [Saninizers](https://github.com/google/Sanitizers)
- CMake 探究可执行命令

## Chapter 06 生成源码

> 大多数项目,使用版本控制跟踪源码; 源代码通常作为构建系统的输入,将其转换为o文件、库或可执行程序; 某些情况下, 使用构建系统在配置或构建步骤时生成源代码, 根据配置步骤中收集的信息, 对源代码进行微调; 另一个常用的方式, 是记录有关配置或编译的信息, 以保证代码行为可重现性.

- CMake 配置时生成源码 | Python 在配置时和构建时生成源码
- CMake 记录项目版本信息 | 从文件中记录项目版本
- [语义化版本](https://semver.org/lang/zh-CN/)
- CMake 配置时和构建时记录 Git Hash 值

## Chapter 07 构建项目

> CMake 如何组合这些构建块,并引入抽象,并最小化代码重复、全局变量、全局状态和显式排序，以免 CMakeLists.txt 文件过于庞大; 目标是为模块化 CMake 代码结构和限制变量范围提供模式, 讨论一些策略,也将帮助控制中大型代码项目的 CMake 代码复杂性.

## Chapter 08 超级构建模式

> 每个项目都需要处理依赖关系,使用 CMake 很容易查询这些依赖关系是否存在于配置项目中;CMake 如何找到安装在系统上的依赖项模式, 当不满足依赖关系, 只能使配置失败, 并向用户警告失败的原因; CMake 可以组织项目如果在系统上找不到依赖项, 就可以自动获取和构建依赖项; 'ExternalProject.cmake' and 'FetchContent.cmake' 标准模块, 在超级构建模式中的如何使用, 前者允许在构建时检索项目的依赖项, 后者允许在配置时检索依赖项; 使用超级构建模式, 可以利用 CMake 作为包管理器: 相同的项目中, 将以相同的方式处理依赖项, 无论依赖项在系统上是已经可用, 还是需要重新构建.

## Chapter 09 语言混合项目

> 有很多的库比较适合特定领域的任务,直接使用这些专业库是一种快捷的方式; 将编译语言代码与解释语言的代码集成在一起, 变得确实越来越普遍, 可以将Python等语言的表达能力与编译语言的性能结合起来, 后者在内存寻址方面效率接近于极致,达到两全其美的目的; [CMake Languages](https://cmake.org/cmake/help/latest/release/3.8.html#languages) : C/C++, CUDA, ASM, Fortran, Java, Python.

## Chapter 10 编写安装程序

> 利用 CMake 安装项目,生成输出头文件,输出目标,安装超级构建; 直到完成安装一个简单的C++项目: 从项目中构建的文件, 并复制到正确的目录, 确保其他项目使用CMake时可以找到该工程的输出目标.

## Chapter 11 打包项目

> CMake 生成源代码和二进制包;CMake/pybind11构建的C++/Python项目,通过PyPI发布;以Conda包的形式发布一个简单的项目; 不同的打包策略, CMake中的工具CPack进行打包,还提供打包和上传CMake项目到Python包索引PyPI和Anaconda云的方法, 这些都是通过包管理器pip和Conda.

## Chapter 12 构建文档

> CMake 利用 Doxygen & Sphinx 构建并生成文档; [Doxygen](https://www.doxygen.nl/)源代码文档工具可以在代码中添加文档标记作为注释, 而后运行Doxygen提取这些注释, 并以Doxyfile配置文件中定义的格式创建文档, Doxygen可以输出HTML、XML，甚至LaTeX或PDF; [Sphinx](https://www.sphinx-doc.org/en/master/), 当与Python项目一起使用时, 可以为 'docstring' 解析源文件, 并自动为函数和类生成文档页面, Sphinx不仅限于Python, 还可以解析Markdown, 并生成HTML, ePUB或PDF文档, 还有[在线阅读服务](https://readthedocs.org), 它提供了一种快速编写和部署文档的方法. 使用[Breathe插件](https://breathe.readthedocs.io/en/latest/)组合 Doxygen 和 Sphinx.

## Chapter 13 选择生成器和交叉编译

> CMake 配置一个项目, 并生成构建工具或框架用于构建项目的文件; CMake包括本地构建工具或集成开发环境(IDE)的生成器.

```shell
# 可以使用 cmake -G 的方式来选择生成器：
cmake -G "Visual Studio 17 2022"
cmake -G "MinGW Makefiles"
cmake -G "Unix Makefiles"

# 不是每个平台上所有的生成器都可用
# CMake 在运行时获取平台信息, 查看当前平台上所有可用生成器的列表
cmake -G
```

## Chapter 14 测试面板

> 将测试部署到 CDash, 用于汇集 CTest 在测试运行期间,夜间测试期间或在持续集成中的测试结果, 面板报告就是 CDash 时.

## Chapter 15 使用 CMake 构建已有项目

> [Vim](https://www.vim.org ) and [Neovim](https://github.com/neovim/neovim) 的源代码, 并尝试将配置和编译, 从 Autotools 迁移到 CMake. 如何开始迁移项目; 生成文件并编写平台检查; 检测所需的依赖关系和链接; 复制编译标志; 移植测试; 移植安装目标; 项目转换为 CMake 的常见问题.
