#include <iostream>

// 抽象基类(Abstract Base Class, ABC)和纯虚函数(pure virtual function)
// 不能实例化的基类被称之为抽象基类，只有一个用途，用于派生
// C++ 通过纯虚函数来实现抽象基类

// 实现抽象基类 ABC
class Fish
{
public:
  // define a pure virtual function Swim
  // pure virtual function must be implemented.
  virtual void Swim() = 0;
};

class Tuna: public Fish
{
public:
  void Swim()
  {
    std::cout << "Tuna swims fast in the sea." << std::endl;
  }
};

class Carp: public Fish
{
public:
  void Swim()
  {
    std::cout << "Carp swims fast in the sea." << std::endl;
  }
};

void MakeFishSwim(Fish& inputFish)
{
  inputFish.Swim();
}


int main(int argc, char** argv)
{
  // 不能实例化抽象基类
  // Fish myFish;
  // 虽然不能实例化抽象基类，可以将指针或者引用的类型指定为抽象基类
  // ABC 提供一种很好的机制，能够声明所有派生类都必须要实现的函数，有助于约束程序的设计
  // 常用于指定派生类的方法名称以及属性，即指定派生类的接口 API

  Carp myLunch;
  Tuna myDenner;

  MakeFishSwim(myLunch);
  MakeFishSwim(myDenner);
  
  return 0;
}

// $ g++ -o main 11.6_abstract_base_class.cpp 
// $ ./main.exe 
// Carp swims fast in the sea.
// Tuna swims fast in the sea.