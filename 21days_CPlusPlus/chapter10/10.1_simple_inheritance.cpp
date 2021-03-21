#include <iostream>

// 继承
class Fish
{
public:
  bool isFreshWateFish;

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

  return 0;
}

// $ g++ -o main 10.1_simple_inheritance.cpp 
// $ ./main.exe 

// About my food:       
// Lunch: Swims in lake.
// Dinner: Swims in sea.