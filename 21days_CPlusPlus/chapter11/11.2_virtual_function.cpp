#include <iostream>

class Fish
{
public:
  // 使用虚函数实现多态行为
  // 使用关键字 virtual 保证编译器调用覆盖版本
  virtual void Swim()
  {
    std::cout << "Fish swim! " << std::endl;
  }
};

class Tuna: public Fish
{
public:
  // override Fish::Swim
  void Swim()
  {
    std::cout << "Tuna swims!" << std::endl;
  }
};

class Carp: public Fish
{
public:
  // override Fish::Swim
  void Swim()
  {
    std::cout << "Carp swims!" << std::endl;
  }
};

// 通过调用基类的虚函数，实现同一个方法展现不同的状态方式
void MakeFishSwim(Fish& inputFish)
{
  // calling virtual function (method) Swim()
  inputFish.Swim();
}


int main(int argc, char** argv)
{
  Tuna myDinner;
  Carp myLunch;

  // 这就是多态 polymorph 
  // 将派生类对象视为基类对象，并执行派生类的方法实现

  // sending Tuna as Fish
  MakeFishSwim(myDinner);

  // sending Carp as Fish
  MakeFishSwim(myLunch);

  return 0;
}

// $ g++ -o main 11.2_virtual_function.cpp 
// $ ./main.exe 
// Tuna swims!
// Carp swims!