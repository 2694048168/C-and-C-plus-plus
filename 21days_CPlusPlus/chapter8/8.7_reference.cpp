#include <iostream>

void GetSequare(int& number)
{
  number *= number;
}

void GetSequare(const int& number, int& result)
{
  result = number * number;
}

int main(int argc, char** argv)
{
  // 1. 引用时对应变量的别名
  int original = 30;
  std::cout << "original = " << original << std::endl;
  std::cout << "original is at address: " << std::hex << &original << std::endl;

  int& ref_original = original;
  std::cout << "===================" << std::endl;
  std::cout << "ref_original = " << ref_original << std::endl;
  std::cout << "ref_original is at address: " << std::hex << &ref_original << std::endl;

  int& reference_original = ref_original;
  std::cout << "===================" << std::endl;
  std::cout << "reference_original = " << reference_original << std::endl;
  std::cout << "reference_original is at address: " << std::hex << &reference_original << std::endl;

  // 2. 将 引用 用于函数参数传递
  std::cout << "Enter a number you wish to square: ";
  int number = 0;
  std::cin >> number;

  GetSequare(number);
  std::cout << "===================" << std::endl;
  std::cout << "Sequare is: " << number << std::endl;

  // 3. 将 const 用于引用
  // 禁止通过引用修改所指向的变量的数据
  // const int& constRef = orginial;

  // 4. 按引用向函数传递参数
  // 避免实参复制形参，提高性能
  std::cout << "Enter a number you wish to square: ";
  number = 0;
  std::cin >> number;
  int result = 0;
  GetSequare(number, result);
  std::cout << "===================" << std::endl;
  std::cout << "Sequare is: " << result << std::endl;

  return 0;
}

// $ g++ -o main 8.7_reference.cpp 
// $ ./main.exe

// original = 30
// original is at address: 0x61fe0c
// ===================
// ref_original = 1e
// ref_original is at address: 0x61fe0c
// ===================
// reference_original = 1e
// reference_original is at address: 0x61fe0c
// Enter a number you wish to square: 2
// ===================
// Sequare is: 4
// Enter a number you wish to square: 2
// ===================
// Sequare is: 4