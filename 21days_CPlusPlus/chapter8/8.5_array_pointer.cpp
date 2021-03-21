#include <iostream>

int main(int argc, char** argv)
{
  // static array of 5 integers.
  const int ARRAY_LEG = 5;
  int numbers[ARRAY_LEG] = {24, -1, 363, -999, 2021};

  // array assigned to pointer to int
  // 数组变量是指向第一个元素的指针(内存地址)
  int* pointerToNums = numbers;

  // Display address contained in pointer
  std::cout << "array[0] == pointer" << std::endl;
  std::cout << "pointerToNums = 0x" << std::hex << pointerToNums << std::endl;

  // Address of first element of array
  std::cout << "&numbers[o] = 0x" << std::hex << &numbers[0] << std::endl;

  std::cout << "=======================================" << std::endl;
  // 使用解引用运算符 * 访问数组中的元素以及将数组运算符 [] 用于指针

  std::cout << "Display array using pointer syntax, operator*" << std::endl;
  for (size_t i = 0; i < ARRAY_LEG; ++i)
  {
    std::cout << "Element " << i << " = " << *(numbers + i) << std::endl;
  }
  
  std::cout << "Display array using ptr with array syntax, operator[]" << std::endl;
  for (size_t i = 0; i < ARRAY_LEG; ++i)
  {
    std::cout << "Element " << i << " = " << pointerToNums[i] << std::endl;
  }

  return 0;
}

// $ g++ -o main 8.5_array_pointer.cpp
// $ ./main.exe

// array[0] == pointer
// pointerToNums = 0x0x61fde0
// &numbers[o] = 0x0x61fde0
// =======================================      
// Display array using pointer syntax, operator*
// Element 0 = 18
// Element 1 = ffffffff
// Element 2 = 16b
// Element 3 = fffffc19
// Element 4 = 7e5
// Display array using ptr with array syntax, operator[]       
// Element 0 = 18
// Element 1 = ffffffff
// Element 2 = 16b
// Element 3 = fffffc19
// Element 4 = 7e5