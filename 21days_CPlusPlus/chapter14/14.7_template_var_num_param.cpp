#include <iostream>

// 参数数量可变的模板
// 编写一个可以计算任意参数数量的和，需要参数数量可变的模板
// 这样使得模板不仅可以处理不同类型的数据，还能处理不同数量的数据
// ... C++ 中使用省略号来表明参数数量可变，可以接受任意参数数量，而且参数类型任意
// C++14 标准才支持，提供新的运算符 sizeof...() 来计算可变参数数量模板传递了多少个参数
// 这样支持了元组

template <typename Res, typename ValType>
void Sum(Res& result, ValType& val)
{
  result = result + val;
}

template <typename Res, typename First, typename... Rest>
void Sum(Res& result, First value1, Rest... valueN)
{
  result = result + value1;
  // 使用递归来反复计算，直到所有参数计算结束
  return Sum(result, valueN ...);
}

int main(int argc, char** argv)
{
  double doubleResult = 0;
  Sum(doubleResult, 3.14, 4.56, 1.112);
  std::cout << "doubleResult = " << doubleResult << std::endl;

  std::string strResult;
  Sum(strResult, "hello ", "world ", "for ", "CPP");
  std::cout << "strResult = " << strResult << std::endl;
  
  return 0;
}

// $ g++ -o main -std=c++14 14.7_template_var_num_param.cpp
// $ ./main.exe
// doubleResult = 8.812
// strResult = hello world for CPP