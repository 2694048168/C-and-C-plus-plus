#include <iostream>
#include <memory> // std::unique_ptr C++11

class Fish
{
public:
  Fish() { std::cout << "Fish: Constructor." << std::endl; }
  ~Fish() { std::cout << "Fish: Destructor." << std::endl; }

  void Swim() const { std::cout << "Fish swims in water." << std::endl; }
};

void MakeFishSwim(const std::unique_ptr<Fish> &smartFishPtr)
{
  smartFishPtr->Swim();
}

int main(int argc, char **argv)
{
  std::unique_ptr<Fish> smartPointerFish(new Fish);

  smartPointerFish->Swim();
  MakeFishSwim(smartPointerFish);

  std::unique_ptr<Fish> copySmartFish;
  // copySmartFish = smartPointerFish;  // error: operator= is private.

  return 0;
}

// $ touch 26.3_unique_ptr.cpp
// $ g++ -o main 26.3_unique_ptr.cpp
// $ ./main.exe

// Fish: Constructor.
// Fish swims in water.
// Fish swims in water.
// Fish: Destructor.