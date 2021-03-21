#include <iostream>

// 参数数量可变的模板
// ... C++ 中使用省略号来表明参数数量可变，可以接受任意参数数量，而且参数类型任意
// C++14 标准才支持，提供新的运算符 sizeof...() 来计算可变参数数量模板传递了多少个参数
void Display()
{
}

template <typename First, typename ...Last> void Display(First value1, Last... valueN)
{
  std::cout << value1 << std::endl;
  Display(valueN...);
}

int main(int argc, char **argv)
{
  Display('a');
  Display(3.14);
  Display('a', 3.14);
  Display('z', 3.14567, "The power of variadic templates!");

  return 0;
}

// $ g++ -std=c++14 -o main 14.12_test_param_num.cpp      
// $ ./main.exe
// a
// 3.14
// a
// 3.14
// z
// 3.14567
// The power of variadic templates!