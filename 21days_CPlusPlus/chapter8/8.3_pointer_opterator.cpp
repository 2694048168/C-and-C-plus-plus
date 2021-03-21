// 指针 进行 自增和自减，其包含的地址增加或者减少的是指向的数据类型的 sizeof，
// 确保编译器将指针指向数据的开头，而不会指向中间或者尾部

// 将 const 用于指针，有三种用法和功效
// 1. 指针本身（内存地址）是常量，不能修改，但是可以修改指针指向的数据
//   int* const ptrType = &ptrData;
// 2. 指针指向的数据是常量，不能修改，但是可以修改指针本身(内存地址)，即该指针可以指向其他内存地址
//   const int* ptrType = &ptrData;
// 3. 指针本身(内存地址)和指针指向数据都是常量，都不能修改
//   const int* const ptrType = &ptrData;

// 指针传递函数时，使用 第三种 最严格的方式，禁止修改指针以及指针指向的数据。

#include <iostream>

int main(int argc, char** argv)
{
  std::cout << "How many integers you wish to enter? ";
  int numEntries = 0;
  std::cin >> numEntries;

  int* ptrToInt = new int [numEntries];

  std::cout << "Allocated for " << numEntries << " integers" << std::endl;
  for (size_t i = 0; i < numEntries; ++i)
  {
    std::cout << "Enter number " << i << ": ";
    std::cin >> *(ptrToInt + i);
  }

  std::cout << "Dispalying all numebers entered: " << std::endl;
  for (size_t i = 0; i < numEntries; ++i)
  {
    std::cout << *(ptrToInt++) << ": ";
  }
  std::cout << std::endl;

  // return pointer to initial position.
  ptrToInt -= numEntries;

  delete[] ptrToInt;

  return 0;
}

// $ g++ -o main 8.3_pointer_opterator.cpp 
// $ ./main.exe 
// How many integers you wish to enter? 3
// Allocated for 3 integers
// Enter number 0: 23
// Enter number 1: 345
// Enter number 2: 2
// Dispalying all numebers entered: 
// 23: 345: 2: