#include <iostream>

class Fish
{
public:
  void Swim()
  {
    std::cout << "Fish swims......" << std::endl;
  }
  // overloaded version
  void Swim(bool FreshWaterFish)
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

// private:
  // accessible only to derived classes
  bool isFreshWaterFish;
};

// 公有继承
class Tuna : public Fish
{
public:
  // 2. 在派生类中使用 using 解除隐藏
  // using Fish::Swim;

  // 3. 覆盖所有重载版本
  // void Swim(bool FreshWaterFish)
  // {
  //   Fish::Swim(isFreshWaterFish);
  // }

  // 重写基类的方法，派生类实例调用的就是派生类的方法
  void Swim()
  {
    std::cout << "Tuna Swims real fast in sea." << std::endl;
  }
};


int main(int argc, char** argv)
{
  Tuna myDinner;

  std::cout << "About my food: " << std::endl;

  std::cout << "Dinner: ";
  myDinner.Swim();
  std::cout << "=============================" << std::endl;
  // myDinner.Swim(false);  // failure hide 在派生类中隐藏基类的方法
  // 覆盖的极端情况就是隐藏，隐藏了所有的重载版本，导致编译错误
  // 要通过派生类的实例调用基类重载版本函数：
  // 1. 使用作用域解析运算符(::)  
  myDinner.Fish::Swim(false);
  // 2. 在派生类中使用 using 解除隐藏
  // 3. 覆盖所有重载版本

  return 0;
}


// $ g++ -o main 10.5_hide_base.cpp 
// $ ./main.exe
// About my food: 
// Dinner: Tuna Swims real fast in sea.
// =============================
// Swims in sea.