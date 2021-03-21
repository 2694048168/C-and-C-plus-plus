#include <iostream>

// constexpr float GetArea() { return PI * radius * radius; }
// constexpr float GetPerimeter() { return 2 * PI * radius; }

int main(int argc, char**argv)
{
  std::cout << "Enter the radius of the circle: ";
  float radius = 0.0;
  std::cin >> radius;

  const float PI = 3.1415926;

  // constexpr float GetArea() { return PI * radius * radius; }
  // constexpr float GetPerimeter() { return 2 * PI * radius; }
  // 不能这样用，常量表达式，必须是可以一次性计算的常量表达式，不能包含变量。

  // float getArea = PI * radius * radius;
  int getArea = PI * radius * radius;
  // float getPerimeter = 2 * PI * radius;
  int getPerimeter = 2 * PI * radius;

  std::cout << "The area of this circle is: " << getArea << std::endl;
  std::cout << "The perimeter of this circle is: " << getPerimeter << std::endl;

  return 0;
}

// $ g++ -o mian 3.10_test_compute_circle.cpp 
// $ ./mian.exe 
// Enter the radius of the circle: 6
// The area of this circle is: 113.097     
// The perimeter of this circle is: 37.6991

// $ g++ -o mian 3.10_test_compute_circle.cpp 
// $ ./mian.exe 
// Enter the radius of the circle: 6
// The area of this circle is: 113    
// The perimeter of this circle is: 37