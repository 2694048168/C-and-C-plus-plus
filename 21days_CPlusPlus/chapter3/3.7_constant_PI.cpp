#include <iostream>

constexpr double GetPI() { return 22.0 / 7; }
constexpr double TwicePI() { return 2 * GetPI(); }

int main(int argc, char**argv)
{
  const double PI = 22.0 / 7;
  // 不建议使用 #define 来定义常量
  std::cout << "The value of constant PI is: " << PI << std::endl;

  // Uncomment next line to view compile failure
  // 常来不可修改值
  // PI = 3.1415926;

  // 使用常量表达式计算 PI 的值
  // constexpr
  std::cout << "constexpr GetPI() returns value: " << GetPI() << std::endl;
  std::cout << "constexpr TwicePI() returns value: " << TwicePI() << std::endl;

  return 0;
}

// $ g++ -o main 3.7_constant_PI.cpp 
// $ ./main.exe 
// The value of constant PI is: 3.14286      
// constexpr GetPI() returns value: 3.14286  
// constexpr TwicePI() returns value: 6.28571