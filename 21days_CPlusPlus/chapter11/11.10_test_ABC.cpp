#include <iostream>

// 抽象基类 ABC
class Shape
{
public:
  virtual double Area() = 0;
  virtual void Print() = 0;
};

// 派生类，必须实现抽象基类中的纯虚函数
class Circle : public Shape
{
public:
  // 利用构造函数进行必要参数的初始化
  Circle(double radius, const double PI) : radius(radius), PI(PI)
  {

  }

  double Area() override
  {
    return PI * radius * radius;
  }

  void Print() override
  {
    std::cout << "This is a Ciecle." << std::endl;
  }

private:
  double radius;
  const double PI = 314159265;
};

// 派生类，必须实现抽象基类中的纯虚函数
class Triangle : public Shape
{
public:
  // 利用构造函数进行必要参数的初始化
  Triangle(const double bottom, const double heigth) : bottom(bottom), heigth(heigth)
  {

  }

  double Area() override
  {
    return 0.5 * bottom * heigth;
  }

  void Print() override
  {
    std::cout << "This is a Triangle." << std::endl;
  }

private:
  const double bottom;
  const double heigth;
};

int main(int argc, char** argv)
{
  double radius = 0;
  double bottom = 0;
  double heigth = 0;
  double PI = 3.1415;

  std::cout << "Please enter the radius of circle: ";
  std::cin >> radius;
  std::cout << "Please enter the value of  PI: ";
  std::cin >> PI;
  std::cout << "Please enter the value of bottom for triangle: ";
  std::cin >> bottom;
  std::cout << "Please enter the value of  height for triangle: ";
  std::cin >> heigth;

  Circle myCircle(radius, PI);
  Triangle myTriangle(bottom, heigth);

  myCircle.Print();
  std::cout << "The area of circle: " << myCircle.Area() << std::endl;

  myTriangle.Print();
  std::cout << "The area of triangle: " << myTriangle.Area() << std::endl;

  return 0;
}

// $ g++ -o main 11.10_test_ABC.cpp
// $ ./main.exe
// Please enter the radius of circle: 3
// Please enter the value of  PI: 3.14159265
// Please enter the value of bottom for triangle: 3
// Please enter the value of  height for triangle: 4
// This is a Ciecle.
// The area of circle: 28.2743
// This is a Triangle.        
// The area of triangle: 6