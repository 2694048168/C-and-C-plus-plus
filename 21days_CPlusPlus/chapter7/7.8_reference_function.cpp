#include <iostream>

const double PI = 3.14159265;

// 使用 引用 进行函数参数的传递，避免大数据的复制，减少开销
// 还可以通过 引用 来实现函数的多个返回值
// output parameter result by reference
void ComputeArea(double radius, double& result)
{
  result = PI * radius * radius;
}

int main(int argc, char** argv)
{
  std::cout << "Enter radius: ";
  double radius = 0;
  std::cin >> radius;

  double areaFetched = 0;
  ComputeArea(radius, areaFetched);

  // Call function to compute area.
  std::cout << "Area is: " << areaFetched << std::endl;

  return 0;
}

// $ g++ -o main 7.8_reference_function.cpp 
// $ ./main.exe 
// Enter radius: 4
// Area is: 50.2655