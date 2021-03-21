#include <iostream>

int main(int argc, char** argv)
{
  int age = 36;
  const double PI = 3.14159265;

  // using & to find the address in memory.
  std::cout << "Integer age is located at: 0X" << &age << std::endl;
  std::cout << "Double PI is located at: 0X" << &PI << std::endl;

  // 使用引用运算符 & 获取变量的地址
  // 指针时指向内存地址的变量，故此使用指针储存内存地址
  std::cout << "==========================" << std::endl;
  
  // 声明并初始化指针
  // pointer initialized to &age.
  int* ptrToAge = &age;

  // Display the value of pointer.
  std::cout << "Integer age is at: 0X" << std::hex << ptrToAge << std::endl;

  std::cout << "==========================" << std::endl;

  // 给指针重新赋值，使其指向另一个变量的地址
  std::cout << "PteToAge points to age now." << std::endl;
  int dogsAge = 9;
  ptrToAge = &dogsAge;
  std::cout << "PtrToAge points to dogsage now." << std::endl;
  std::cout << "ptrToAge is at: 0X" << std::hex << ptrToAge << std::endl;

  std::cout << "==========================" << std::endl;
  
  // 使用 解引用运算符 * 访问指针指向的数据（即内存地址里储存的数据）
  std::cout << "age is at: 0X" << std::hex << &age << std::endl;
  std::cout << "dongsAge is at: 0X" << std::hex << &dogsAge << std::endl;
  std::cout << "ptrToAge value: " << std::dec << *ptrToAge << std::endl;

  // 使用指针变量来储存数据在内存中的地址
  // 通过指针和解引用运算符来操纵数据
  std::cout << "==========================" << std::endl;

  // 将 sizeof() 用于 指针 的结果
  std::cout << "sizeof fundamental types -" << std::endl;
  std::cout << "sizeof(char) = " << sizeof(char) << std::endl;
  std::cout << "sizeof(int) = " << sizeof(int) << std::endl;
  std::cout << "sizeof(double) = " << sizeof(double) << std::endl;

  std::cout << "sizeof pointers to fundamental types -" << std::endl;
  std::cout << "sizeof(char*) = " << sizeof(char*) << std::endl;
  std::cout << "sizeof(int*) = " << sizeof(int*) << std::endl;
  std::cout << "sizeof(double*) = " << sizeof(double*) << std::endl;

  return 0;
}

// $ g++ -o main 8.1_get_variable_addr.cpp 
// $ ./main.exe

// Integer age is located at: 0X0x61fe14
// Double PI is located at: 0X0x61fe08  
// ==========================
// Integer age is at: 0X0x61fe14        
// ==========================
// PteToAge points to age now.
// PtrToAge points to dogsage now.      
// ptrToAge is at: 0X0x61fe04
// ==========================
// age is at: 0X0x61fe14
// dongsAge is at: 0X0x61fe04
// ptrToAge value: 9
// sizeof fundamental types -
// sizeof(char) = 1
// sizeof(int) = 4
// sizeof(double) = 8
// sizeof pointers to fundamental types -
// sizeof(char*) = 8
// sizeof(int*) = 8
// sizeof(double*) = 8