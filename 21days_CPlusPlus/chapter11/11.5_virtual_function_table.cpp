#include <iostream>

// 虚函数工作原理 —— 理解虚函数表
// 编译阶段，编译器不知道传递的对象时哪一种
// 具体的对象是在运行时候决定的，使用多态的逻辑实现
// 而这种逻辑是在编译阶段提供的
// 编译器将为实现了虚函数的类创建虚函数表 virtual functino table VFT
// 同时创建隐藏的指针 VFT* ，可以看作一个包含函数指针的静态数组，每个指针指向相应的虚函数
// 虚函数表就是这样帮助实现 C++ 多态的

// $ g++ -o main 11.5_virtual_function_table.cpp 
// $ ./main.exe 
// sizeof(SimpleClass) = 8
// sizeof(Base) = 16 

class SimpleClass
{
public:
  void DoSomething(){}

private:
  int a, b;
};


class Base
{
public:
  virtual void DoSomething(){}

private:
  int a, b;
};

int main(int argc, char** argv)
{

  std::cout << "sizeof(SimpleClass) = " << sizeof(SimpleClass) << std::endl;
  std::cout << "sizeof(Base) = " << sizeof(Base) << std::endl;
  
  return 0;
}
