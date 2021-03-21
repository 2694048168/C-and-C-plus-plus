#include <iostream>

const double PI = 3.14159265;

// using auto of Function std==C++14
// 注意 auto 的一些特性要求，推断的前提条件！！！
auto ComputeArea(double radius)
{
  return PI * radius * radius;
}

int main(int argc, char** argv)
{
  std::cout << "Enter radius: ";
  double radius = 0;
  std::cin >> radius;

  // Call function to compute area.
  std::cout << "Area is: " << ComputeArea(radius) << std::endl;

  return 0;
}
