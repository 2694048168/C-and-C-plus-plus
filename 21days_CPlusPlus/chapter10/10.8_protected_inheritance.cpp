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

// 保护继承
// has-a 关系
// 派生类能够访问基类的所有公共和保护成员
// 在继承层次结构外，不能通过派生类实例访问基类的公有成员
// 当且仅当必要的时候，才采用私有或者保护继承，否则会出现兼容性瓶颈
class Car: protected Motor
{
public:
  void Move()
  {
    SwitchIgnition();
    PumpFuel();
    FireCylinders();
  }
};

// 保护继承
class RaceCar: protected Car
{
public:
  void Move()
  {
    SwitchIgnition();
    PumpFuel();
    FireCylinders();
    FireCylinders();
    FireCylinders();
  }
};

int main(int argc, char** argv)
{
  RaceCar myDereamCar;
  myDereamCar.Move();

  // 切除问题
  // 派生类对象复制给基类对象，无论通过显示复制，还是参数传递，
  // 编译器都只是复制部分数据，这种无意间的裁剪数据，导致派生类变为基类的行为称之为切除 slicing
  // 避免 slicing 的方法就是，不要以值传递的方式传递参数，
  // 而是使用指向基类的指针或者const引用的方式传递

  return 0;
}

// $ g++ -o main 10.8_protected_inheritance.cpp 
// $ ./main.exe 
// Ignition ON      
// Fuel in cylinders
// Vroooom
// Vroooom
// Vroooom