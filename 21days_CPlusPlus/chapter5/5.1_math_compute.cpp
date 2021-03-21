#include <iostream>

int main(int argc, char** argv)
{
  // 编译器会忽略空白：空格、制表符、换行符、回车等
  // 但是在字符串字面量中的空白将导致输出不同
  // 语句、语句块、运算符、算术运算和逻辑运算
  // 左值：通常是内存地址；右值：是内存单元里面的内容

  std::cout << "Enter two integers: " << std::endl;
  int numOne = 0, numTwo = 0;
  std::cin >> numOne;
  std::cin >> numTwo;

  std::cout << numOne << " + " << numTwo << " = " << numOne + numTwo << std::endl;
  std::cout << numOne << " - " << numTwo << " = " << numOne - numTwo << std::endl;
  std::cout << numOne << " * " << numTwo << " = " << numOne * numTwo << std::endl;
  std::cout << numOne << " / " << numTwo << " = " << numOne / numTwo << std::endl;
  std::cout << numOne << " % " << numTwo << " = " << numOne % numTwo << std::endl;
  // Postfix increment operator and Postfix decrement operator
  std::cout << numOne << " Postfix increment " << numOne++ << std::endl;
  std::cout << numOne << " Postfix decrement " << numOne-- << std::endl;
  // Prefix increment operator and Prefix decrement operator
  std::cout << numTwo << " Prefix increment " << ++numTwo << std::endl;
  std::cout << numTwo << " Prefix decrement " << --numTwo << std::endl;

  // 后缀：先将右值赋值到左值，右值再执行递增或者递减；这说明需要一个额外的临时变量开销储存右值
  // 前缀：先将右值执行递增或者递减，再将结果赋值到左值，不需要临时开销，建议使用前缀
  // 注意区分前缀和后缀运算符是否对计算结果有所影响！！！

  return 0;
}

// $ g++ -o mian 5.1_math_compute.cpp 
// $ ./mian.exe 
// Enter two integers: 
// 32
// 32
// 32 + 32 = 64
// 32 - 32 = 0
// 32 * 32 = 1024
// 32 / 32 = 1
// 32 % 32 = 0
// 32 Postfix increment 32
// 33 Postfix decrement 33
// 32 Prefix increment 33 
// 33 Prefix decrement 32