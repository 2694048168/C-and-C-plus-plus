#include <iostream>

class Motor
{
public:
  void SwitchIgnition()
  {
    std::cout << "Ignition ON" << std::endl;
  }

  void PumpFuel()
  {
    std::cout << "Fuel in cylinders" << std::endl;
  }

  void FireCylinders()
  {
    std::cout << "Vroooom" << std::endl;
  }
};

// 私有继承
// 私有继承意味着派生类的实例中，基类的所有公有成员和方法都是私有的，不能从外部访问
// 即使是基类的公有成员和方法，只能被派生类使用，无法通过派生类的实例来使用
class Car: private Motor
{
public:
  void Move()
  {
    SwitchIgnition();
    PumpFuel();
    FireCylinders();
  }
};


int main(int argc, char** argv)
{
  Car myDereamCar;
  // myDereamCar.PumpFuel();  // failure!!!
  myDereamCar.Move();

  return 0;
}

// $ g++ -o main 10.7_private_inheritance.cpp 
// $ ./main.exe 
// Ignition ON      
// Fuel in cylinders
// Vroooom