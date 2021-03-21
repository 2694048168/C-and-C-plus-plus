#include <iostream>

class Fish
{
public:
  // overload constructor
  Fish(bool isFreshWater) : isFreshWaterFish(isFreshWater)
  {

  }

  void Swim()
  {
    if (isFreshWaterFish)
    {
      std::cout << "Swims in lake." << std::endl;
    }
    else
    {
      std::cout << "Swims in sea." << std::endl;
    }
  }

private:
  // accessible only to derived classes
  bool isFreshWaterFish;
};

// 公有继承
class Tuna : public Fish
{
public:
  // constructor initializes base
  Tuna() : Fish(false)
  {

  }

  // 重写基类的方法，派生类实例调用的就是派生类的方法
  void Swim()
  {
      std::cout << "Tuna Swims real fast in sea." << std::endl;
  }
};

// 公有继承
class Carp : public Fish
{
public:
  // constructor initializes base
  Carp() : Fish(true)
  {

  }

  // 重写基类的方法，派生类实例调用的就是派生类的方法
  void Swim()
  {
    std::cout << "Carp Swims real slow in lake." << std::endl;
  }
};


int main(int argc, char** argv)
{
  Carp myLunch;
  Tuna myDinner;

  std::cout << "About my food: " << std::endl;

  std::cout << "Lunch: ";
  // myLunch.Swim(); 调用的是派生类重写的方法
  myLunch.Swim();

  std::cout << "Dinner: ";
  // myDinner.Swim();  调用的是派生类重写的方法
  myDinner.Swim();

  // 调用基类中被覆盖的方法，即在派生类实例中调用基类被覆盖的方法
  // 需要使用 作用域解析运算符 :: 来调用基类方法

  std::cout << "===========================" << std::endl;
  std::cout << "Lunch: ";
  // 通过作用域解析运算符进行显示调用基类方法
  myLunch.Fish::Swim();

  std::cout << "Dinner: ";
  // 通过作用域解析运算符进行显示调用基类方法
  myDinner.Fish::Swim();

  return 0;
}


// $ g++ -o main 10.4_cover_base.cpp 
// $ ./main.exe 
// About my food:
// Lunch: Carp Swims real slow in lake.
// Dinner: Tuna Swims real fast in sea.
// ===========================
// Lunch: Swims in lake.
// Dinner: Swims in sea.