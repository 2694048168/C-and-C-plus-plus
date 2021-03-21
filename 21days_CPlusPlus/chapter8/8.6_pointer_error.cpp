#include <iostream>

// 使用指针时常见编程错误
// 1. 内存泄漏
//   运行时间长，占用内存越多，系统越慢；new and delete must together.
// 2. 指针指向无效的内存地址
//   使用解引用运算符之前必须保证指针有效；内存管理很重要
// 3. 悬浮指针（迷途指针 或者 失控指针 或者 野指针）
//   delete 释放之后 指针无效，应该设置为 NULL
// 4. 检查 new 的内存分配申请是否得到满足
// 异常机制，new 不一定成功，要看内存占用情况

int main(int argc, char** argv)
{
  std::cout << "Is it sunny (y/n)? ";
  char userInput = 'y';
  std::cin >> userInput;

  // declare pointer and initialize
  bool* const isSunny = new bool;

  *isSunny = true;

  if (userInput == 'n')
  *isSunny = false;

  std::cout << "Boolean flag sunny says: " << *isSunny << std::endl;

  // release valid memory
  delete isSunny;


  // 使用 new(nothrow) 在分配内存失败时返回 NULL
  int* ptrToManyNums = new(std::nothrow) int [0x1fffffff];
  // check ptrToManyNums != NULL
  if (ptrToManyNums)
  {
    std::cout << "Memory allocation succesfully." << std::endl;
    delete[] ptrToManyNums;
  }
  else
  {
    std::cout << "Memory allocation failed." << std::endl;
  }
  
  return 0;
}