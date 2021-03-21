#include <iostream>

// 将 const 用于指针，有三种用法和功效
// 1. 指针本身（内存地址）是常量，不能修改，但是可以修改指针指向的数据
//   int* const ptrType = &ptrData;
// 2. 指针指向的数据是常量，不能修改，但是可以修改指针本身(内存地址)，即该指针可以指向其他内存地址
//   const int* ptrType = &ptrData;
// 3. 指针本身(内存地址)和指针指向数据都是常量，都不能修改
//   const int* const ptrType = &ptrData;

// 指针传递函数时，使用 第三种 最严格的方式，禁止修改指针以及指针指向的数据。
void CalcArea(const double* const ptrPI, const double* const ptrRadius, double* const ptrArea)
{
  // check pointers for validity before using.
  if (ptrPI && ptrRadius && ptrArea)
  {
    *ptrArea = (*ptrPI) * (*ptrRadius) * (*ptrRadius);
  }
}

int main(int argc, char** argv)
{
  const double PI = 3.14159265;
  std::cout << "Enter radius of circle: ";
  double radius = 0;
  std::cin >> radius;

  double area = 0;
  CalcArea(&PI, &radius, &area);

  std::cout << "Area is: " << area << std::endl;

  return 0;
}

// $ g++ -o main 8.4_pointer_function.cpp 
// $ ./main.exe 
// Enter radius of circle: 43
// Area is: 5808.8