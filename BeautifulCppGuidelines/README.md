## Beautiful C++ 

> Core Guidelines for Writing Clean, Safe and Fast Code

> What I cannot create, I do not understand. Know how to solve every problem that has been solved. Quote from Richard Feynman


### **Features**
- [x] The C++ Core Guidelines(CG) presents its rules relatively tersely in a fixed format
- [x] The C++ Core Guidelines(CG) rules are often expressed in language-technical terms with an emphasis on enforcement through static analysis
- [x] The guidelines provide excellent, simple advice for improving your C++ style such that you can write correct, performant, and efficient code at your first attempt
- [x] [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- [x] **Write in ISO Standard C++** and **Extension to C++**,使用ISO标准C++,而不使用C++编译器扩展

```shell
# 只启用 ISO C++ 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 只启用 ISO C 标准的编译器标志, 而不使用特定编译器的扩展
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
```

- [x] **Use #include guards for all header files, NOT #pragma once**,使用header guards防止头文件包含,减少编译时间

```C++
// 现代编译器基本都支持该宏定义
#pragma once

// header guards
#ifndef __HEADER_GUARDS_H__
#define __HEADER_GUARDS_H__

void some_func(char str[]);

#endif /* __HEADER_GUARDS_H__ */

```

- [x] Modern C++11, new types were introduced in header <cstdint> that **defined fixed-width integral types**
- [x] **Backward compatibility Forward compatibility** and "Y2K"
- [x] [ISO CPP](https://isocpp.org/) and Cpp Conferences(CppCon)
- [x] **Prefer default arguments over overloading**, 抽象接口API时候优先使用默认参数而非重载
- [x] 不要定义仅初始化数据成员的默认构造函数而应使用类成员初始化
- [x] 避免平凡的get和set函数
- [x] 每条语句只声明一个名字
- [x] 不强求函数只用一条return语句
- [x] 将凌乱的结构封装起来，而不是使其散布于代码中
- [x] 尽量减少函数参数
- [x] 使用C风格子集获取跨编译器的ABI
- [x] 按成员声明顺序定义并初始化成员变量
- [x] 尽量减少可写数据的显式共享
- [x] 只在真正需要时使用模板元编程
- [x] 切勿通过原生指针（T*）或引用（T&）转移所有权
- [x] 避免使用单例
- [x] 依靠构造函数和赋值运算符，而不是 memset 和 memcpy
- [x] 不要用强制转换去除const限定符
- [x] 避免基于全局状态（如errno）的错误处理
- [x] 不要在头文件的全局作用域写using namespace
- [x] 优先选择结构体或元组返回多个“输出”值
- [x] 优先选择类枚举而不是“普通”枚举
- [x] 保持作用域小
- [x] 使用constexpr表示编译时可以计算的值
- [x] 使用模板提高代码的抽象层次
- [x] 为所有模板参数指定概念
- [x] 理想情况下，程序应具有静态类型安全性
- [x] 优先选择不可变数据而不是可变数据
- [x] 封装违反规则的部分
- [x] 确定初始值后再声明变量
- [x] 为促成优化而设计
- [x] 使用RAII防止泄露
