// 微处理器如何处理函数调用
// 指令栈 —— 指针
// 内联函数 inline，编译器视为申请或者请求，不一定成功
// 可能导致代码急剧膨胀，当且仅当函数非常简单，需要降低开销时采用

#include <iostream>

// define inline function
inline long DoubleNum(int inputNum)
{
  return inputNum * 2;
}

int main(int argc, char** argv)
{
  std::cout << "Enter an integer: ";
  int inputNUm = 0;
  std::cin >> inputNUm;

  // Call inline function
  std::cout << "Double is: " << DoubleNum(inputNUm) << std::endl;

  return 0;
}