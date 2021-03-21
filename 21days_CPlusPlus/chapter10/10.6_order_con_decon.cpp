#include <iostream>

class FishDummyMember
{
public:
  // default constructor
  FishDummyMember()
  {
    std::cout << "FishDummyMember constructor" << std::endl;
  }

  // default destructor
  ~FishDummyMember()
  {
    std::cout << "FishDummyMember destructor" << std::endl;
  }
};

class Fish
{
public:
  // Fish default constructor
  Fish()
  {
    std::cout << "Fish constructor" << std::endl;
  }

  // Fish default destructor
  ~Fish()
  {
    std::cout << "Fish destructor" << std::endl;
  }

protected:
  FishDummyMember dummy;
};

class TunaDummyMember
{
public:
  // default constructor
  TunaDummyMember(/* args */)
  {
    std::cout << "TunaDummyMember constructor" << std::endl;
  }

  // default destructor
  ~TunaDummyMember()
  {
    std::cout << "TunaDummyMenber destructor" << std::endl;
  }
};

class Tuna: public Fish
{
public:
  Tuna(/* args */)
  {
    std::cout << "Tuna constructor" << std::endl;
  }

  ~Tuna()
  {
    std::cout << "Tuna destructor" << std::endl;
  }
};

int main(int argc, char** argv)
{
  Tuna myDinner;
  //基类和派生类的 构造 和 析构 的顺序
  // 1. 首先构造没有继承体系的类
  // 2. 然后构造基类
  // 3. 最后构造派生类

  // 4. 首先析构派生类
  // 5. 然后析构基类
  // 6. 最后析构没有继承体系的类

  return 0;
}

// $ g++ -o main 10.6_order_con_decon.cpp 
// admin@weili /d/VSCode/workspace/21days_CPlusPlus/chapter10
// $ ./main.exe 
// FishDummyMember constructor
// Fish constructor
// Tuna constructor
// Tuna destructor
// Fish destructor
// FishDummyMember destructor 