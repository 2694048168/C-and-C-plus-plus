## Protocol Buffers

**Protocol Buffers are language-neutral, platform-neutral extensible mechanisms for serializing structured data.**

> Protocol Buffers（简称：ProtoBuf）是一种开源跨平台的**序列化数据结构**的协议。其对于存储资料或在网络上进行通信的程序是很有用的。这个方法包含一个**接口描述语言**，描述一些数据结构，并提供程序工具根据这些描述产生代码，这些代码将用来生成或解析代表这些数据结构的**字节流**。Google最初开发了Protocol Buffers用于内部使用, 设计目标是简单和性能, 被设计地与XML相比更小且更快, 被广泛用来存储和交换各种类型的结构化数据, 可以被当作一个RPC系统的基础，并被用于几乎所有的跨服务器通信。proto2提供一个程序产生器，支持C++、Java和Python；proto3提供一个程序产生器，支持C++、Java (包含JavaNano)、Python、Go、Ruby、Objective-C、C#、JavaScript。

**Protobuf不建议使用3.21.11以上的版本,新版本比旧版本多了一个abseil的三方库,且安装方式比较复杂**
- [Protocol Buffers v3.21.12 download](https://github.com/protocolbuffers/protobuf/releases/tag/v21.12)

```shell
# https://github.com/protocolbuffers/protobuf/blob/main/cmake/README.md
# https://github.com/abseil/abseil-cpp
# cmake build and install  Abseil installed 

# https://github.com/protocolbuffers/protobuf/releases/latest
cd protobuf
mkdir install

cmake -S. -B build -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_CXX_STANDARD=17 -Dprotobuf_ABSL_PROVIDER=package  -DCMAKE_PREFIX_PATH=D:/DevelopTools/abseil-cpp-20240116.2/install/lib/cmake/absl # Path to where I installed Abseil

# Windows open VS to build and install
cmake --build build --config --config Release
cmake --build build --config --config Debug

cmake --install build
cmake --install build --prefix D:/DevelopTools

```

### Protocol Buffers 标准使用流程

![workflow](https://protobuf.dev/images/protocol-buffers-concepts.png)

- Protocol Buffers 常用于网络传输(TCP/IP), 配置文件, 数据存储(image)
- 对比Json和XML格式优势, 二进制流效率和速度更快, 跨平台跨编程语言
- C++ programmers introduction to working with protocol buffers
    - Define message formats in a **.proto** file
    - Use the protocol buffer **compiler**
    - Use the C++ protocol buffer API to **write** and **read** messages
- basic **data type** in Protobuf and in C++
- message and enum in Protobuf **VS** struct and enum in C++
- repeated 限定修饰符 in Protobuf **VS** array in C++
- import and package in Protobuf **VS** include and namespace in C++
- 序列化 and 反序列化 via Protobuf in C++
- build and compile protocol buffers on Windows and Linux
- the command-line tool: protoc

```shell
mkdir ConfParams
touch ConfParams/Person.proto
touch ConfParams/Address.proto

protoc --version

# protoc -I path .proto文件 --cpp_out=输出路径(存储生成的c++文件)
protoc ./ConfParams/Address.proto --cpp_out=.
protoc ./ConfParams/Person.proto --cpp_out=.
# 或者使用 -I 参数
protoc -I ./ConfParams Person.proto --cpp_out=.

# generate the "Address.pb.h" and "Address.pb.cc" 
# generate the "Person.pb.h" and "Person.pb.cc" 

mkdir src
touch src/main.cpp

```

### C++ proto CMakeLists 构建
> 在C++项目中使用Protocol Buffers时, 在CMakeLists.txt 文件中适当地配置以编译.proto文件, 并链接生成的C++代码.

1. 查找并包含Protobuf编译器和库：首先，需要确保CMake能够找到Protobuf编译器（protoc）和Protobuf库
2. 定义.proto文件变量：指定需要编译的.proto文件
3. 使用protobuf_generate_cpp宏：这个宏会自动运行protoc编译器生成C++代码
4. 包含生成的头文件：包含自动生成的头文件
5. 添加可执行文件或库：使用add_executable或add_library命令添加你的可执行文件或库，并链接Protobuf库
6. 链接Protobuf库：确保链接Protobuf编译器生成的对象文件和Protobuf库

**NOTE**: [.proto 嵌套使用问题](https://oldpan.me/archives/protobuf-cmake-right-usage)
**NOTE**: [VS 属性配置自定义生成工具](https://blog.csdn.net/oLuoJinFanHua12/article/details/104993853)

```shell
# 修改以下属性：
# 1. 命令行：
$(SolutionDir)tools\protoc.exe -I .\proto %(Filename).proto --cpp_out=$(ProjectDir)protocpp

# 2. 说明： 
protoc %(Filename).proto
# 3. 输出： 
$(ProjectDir)protocpp%(Filename).pb.cc
# 4. 向项类型添加输出： 
选择 C/C++ 编译器
```


```C++
// ========== 序列化 ==========
// 头文件目录: google\protobuf\message_lite.h
// --- 将序列化的数据 数据保存到内存中
// 将类对象中的数据序列化为字符串, c++ 风格的字符串, 参数是一个传出参数
bool SerializeToString(std::string* output) const;
// 将类对象中的数据序列化为字符串, c 风格的字符串, 参数 data 是一个传出参数
bool SerializeToArray(void* data, int size) const;

// ------ 写磁盘文件, 只需要调用这个函数, 数据自动被写入到磁盘文件中
// -- 需要提供流对象/文件描述符关联一个磁盘文件
// 将数据序列化写入到磁盘文件中, c++ 风格
// ostream 子类 ofstream -> 写文件
bool SerializeToOstream(std::ostream* output) const;
// 将数据序列化写入到磁盘文件中, c 风格
bool SerializeToFileDescriptor(int file_descriptor) const;

// ========== 反序列化 ==========
// 头文件目录: google\protobuf\message_lite.h
bool ParseFromString(const std::string& data) ;
bool ParseFromArray(const void* data, int size);
// istream -> 子类 ifstream -> 读操作
// wo ri
// w->写 o: ofstream , r->读 i: ifstream
bool ParseFromIstream(std::istream* input);
bool ParseFromFileDescriptor(int file_descriptor);

```

### Use Reference
- [Protocol Buffers GitHub](https://github.com/protocolbuffers/protobuf)
- [Protocol Buffers Documentation](https://protobuf.dev/)
- [VSCode extension Protocol Buffers](https://marketplace.visualstudio.com/items?itemName=zxh404.vscode-proto3)
- [protobuf中的数据类型 和 C++ 数据类型对照表](https://subingwen.cn/cpp/protobuf/)
- [Protocol Buffer Basics: C++](https://protobuf.dev/getting-started/cpptutorial/)
