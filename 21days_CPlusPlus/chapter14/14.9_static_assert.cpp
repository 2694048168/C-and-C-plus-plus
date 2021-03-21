#include <iostream>

// 在实际中 C++ 编程中使用模板
// 模板一个重要而且强大的应用是在标准模板库(STL)中
// STL 由一系列的模板类和函数组成,包含泛型实用类和算法
// 利用 STL 编写出高效的 C++ 程序,有助于在模板细节上浪费时间

// 使用 static_assert 执行编译阶段检查
// static_assert 是 std=C++11 新增的一项功能， 
// 让您能够在不满足指定条件时禁止编译，
// 特别对模板类来说很有用。
// static_assert，它是一种编译阶段断言，可用于在开发环境（或控制台中）显示一条自定义消息：
// static_assert(expression being validated, "Error message when check fails");

// 使用 static_assert 在针对 int 类型实例化时发出抗议
template <typename T>
class EverythingButInt
{
public:
  EverythingButInt()
  {
    static_assert(sizeof(T) != sizeof(int), "No int please!");
  }
};

// $ g++ -std=c++11 -o main 14.9_static_assert.cpp        
// 14.9_static_assert.cpp: In instantiation of 'EverythingButInt<T>::EverythingButInt() [with T = int]':
// 14.9_static_assert.cpp:28:25:   required from here     
// 14.9_static_assert.cpp:17:29: error: static assertion failed: No int please!
//      static_assert(sizeof(T) != sizeof(int), "No int please!");
// 没有输出，因为这个程序不能通过编译，它显示一条错误消息，指出您指定的类型不正确：

int main(int argc, char** argv)
{
  EverythingButInt<int> test; // template instantiation with int.
  // EverythingButInt<double> test; // compile successfully !!!

  return 0;
}
