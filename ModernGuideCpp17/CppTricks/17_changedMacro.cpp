/**
 * @file 17_changedMacro.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * @brief 在现代C++编程中,宏（Macro）的使用逐渐减少,
 * 因为C++标准库和语言本身提供了更多安全、灵活的替代方案
 * 尽管宏在某些特定场景下（如编译时条件、日志管理、代码生成等）仍然有其用武之地
 * 但在大多数情况下, 宏的功能可以被更现代、更安全的机制所取代.
 * 
 * 1. 使用内联函数取代宏
 * 2. 使用constexpr取代宏
 * 3. 使用模板取代宏
 * 4. 使用常量定义取代宏
 * 5. 使用std::function和std::bind取代宏
 * 6. 使用类型萃取（Type Traits）取代宏
 * 
 */

#include <functional>
#include <iostream>
#include <type_traits>

/* 宏的主要目的之一是避免函数调用的开销,
 在现代编译器中, 内联函数（Inline Function）会自动将函数的代码插入到调用点,
 从而消除函数调用的开销. 因此 宏中的代码可以被封装为一个内联函数
 并在需要的地方直接调用该函数 */
#define SQUARE(x) ((x) * (x))

inline int square(int x)
{
    return x * x;
}

/* C++11引入了constexpr关键字, 它可以用于声明常量表达式;
 通过使用constexpr变量或函数,
 可以实现与宏相似的功能,同时具有类型检查和编译期计算的优势 */
#define PI 3.1415926

constexpr double pi = 3.1415926;

/* 宏通常用于定义常量值或一些简单的表达式.
在现代C++中, 可以使用const或constexpr关键字来定义常量
以提高代码的可读性和类型安全性
常量定义不仅具有与宏相似的功能,还提供了类型检查和调试支持,使代码更加安全、易于维护 */
#define MAX_SIZE 100

const int     maxSize  = 100;
constexpr int maxSize_ = 100;

/* C++是一种支持模板和泛型编程的语言,
通过使用模板或泛型编程技术,可以更灵活地处理不同类型的数据,
并避免宏中的类型不安全问题 */
#define MAX(a, b) ((a) > (b) ? (a) : (b))

template<typename T>
T max(T a, T b)
{
    return (a > b) ? a : b;
}

/* 宏有时用于实现复杂的回调机制,
在现代C++中,可以使用std::function和std::bind来实现类似的功能,
同时提供更高的灵活性和类型安全性 */
#define CALLBACK(func, arg) func(arg)

void myCallback(int x)
{
    // 处理回调
    std::cout << "the value is " << x << '\n';
}

std::function<void(int)> callback = std::bind(myCallback, std::placeholders::_1);

/* 宏有时用于进行类型检测和条件编译,
在现代C++中,可以使用类型萃取(Type Traits)来实现类似的功能
同时提供更高的灵活性和类型安全性 */
#define IS_POINTER(type) (std::is_pointer<type>::value)

template<typename T>
bool isPointer()
{
    return std::is_pointer<T>::value;
}

// --------------------------------------
int main(int argc, const char *argv[])
{
    // 使用回调
    callback(10);

    // 使用类型萃取
    bool isPtr = isPointer<int *>();
    if (isPtr)
        std::cout << "The Type via 'Type Traits' is Pointer\n";
    else
        std::cout << "The Type via 'Type Traits' is NOT Pointer\n";

    return 0;
}
