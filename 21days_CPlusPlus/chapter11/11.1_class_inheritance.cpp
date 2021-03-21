#include <iostream>

// 多态 polymorph 是面向对象的特性
// 以类似的方式处理不同类似的对象
// C++ 中通过继承层次结构来实现

class Fish
{
public:
  void Swim()
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

void MakeFishSwim(Fish& inputFish)
{
  // calling Fish::Swim
  inputFish.Swim();
}


int main(int argc, char** argv)
{
  Tuna myDinner;
  // calling Tuna::Swim
  myDinner.Swim();

  // sending Tuna as Fish
  MakeFishSwim(myDinner);
  // 并没有得到想要的预期结果

  return 0;
}

// $ g++ -o main 11.1_class_inheritance.cpp 
// $ ./main.exe 
// Tuna swims!
// Fish swim! 
