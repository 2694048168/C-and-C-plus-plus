#include <iostream>

class CalcCircle
{
public:
  CalcCircle (double radius) : radius(radius), PI(3.14159265) {}

  double CalcCircumference()
  {
    return 2 * PI * radius;
  }

  double CalcArea()
  {
    return PI * radius * radius;
  }

private:
  const double PI;
  double radius;
};

int main()
{
  std::cout << "Please enter the radius of circle: ";
  double radius = 0;
  std::cin >> radius;

  CalcCircle myCircle(radius);
  std::cout << "Circumference fo the circle = " << myCircle.CalcCircumference() << std::endl;
  std::cout << "Area fo the circle = " << myCircle.CalcArea() << std::endl;

  return 0;
}

// $ g++ -o mian 9.18_test_class.cpp 
// $ ./mian.exe
// Please enter the radius of circle: 44
// Circumference fo the circle = 276.46
// Area fo the circle = 6082.12