#include <iostream>

// 继承
class Fish
{
public:
  void Swim()
  {
    if (isFreshWateFish)
    {
      std::cout << "Swims in lake." << std::endl;
    }
    else
    {
      std::cout << "Swims in sea." << std::endl;
    }
  }

  // 基类某些属性能够在派生类中访问，不能在继承层次结构外部访问
protected:
  bool isFreshWateFish;
};

// 公有继承
class Tuna : public Fish
{
public:
  Tuna()
  {
    isFreshWateFish = false;
  }
};

// 公有继承
class Carp : public Fish
{
public:
  Carp()
  {
    isFreshWateFish = true;
  }
};


int main(int argc, char** argv)
{
  Carp myLunch;
  Tuna myDinner;

  std::cout << "About my food: " << std::endl;

  std::cout << "Lunch: ";
  myLunch.Swim();

  std::cout << "Dinner: ";
  myDinner.Swim();

  // uncomment line below to see that protected members are not accessible from outside the class hierarchy
  // myLunch.isFreshWaterFish = false;

  return 0;
}

// $ g++ -o main 10.1_simple_inheritance.cpp 
// $ ./main.exe 

// About my food:       
// Lunch: Swims in lake.
// Dinner: Swims in sea.