#include <iostream>

// test 练习 1
#define MULTIPLE(a, b) ((a) * (b))

// 使用模板实现该宏功能
// #define QUARTER(x) (x / 4)
template <typename T>
T QUARTER(const T value)
{
  return ((value) / 4);
}

// 实现模板函数 swap 交换两个变量的值
template <typename T1, typename T2=T1>
void swap(T1& value1, T2& value2)
{
  T1 temp = value1;
  value1 = value2;
  value2 = temp;
}


int main(int argc, char** argv)
{
  // test 1.
  float num1 = 3, num2 = 8;
  std::cout << "The result of multiple " << MULTIPLE(num1, num2) << std::endl;
  std::cout << "=========================" << std::endl;

  // test 2.
  double value = 16;
  std::cout << "The result of QUARTER: " << QUARTER(value) << std::endl;

  // test swap templeate function  
  std::cout << "=========================" << std::endl;
  int value1 = 2020, value2 = 2021;
  std::cout << "Before swap, the value of number1 and number2: " 
            << value1 << " " << value2 << std::endl; 
  // swap template function
  swap(value1, value2);
  std::cout << "After swap, the value of number1 and number2: " 
            << value1 << " " << value2 << std::endl;
  
  return 0;
}

// $ g++ -o main 14.10_test.cpp 
// $ ./main.exe
// The result of multiple 24
// =========================
// The result of QUARTER: 4
// =========================
// Before swap, the value of number1 and number2: 2020 2021
// After swap, the value of number1 and number2: 2021 2020
