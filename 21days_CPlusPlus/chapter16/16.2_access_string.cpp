#include <iostream>

// 访问 std::string 字符内容
// 1. 使用迭代器
// 2. 使用下标运算符 [] ，提供偏移量，不能越界访问
// 3. 转换为 C-style 字符串，使用成员函数 c_str()

int main(int argc, char** argv)
{
  std::string str1String ("Hello String.");

  // access the contents of the string using array syntax.
  std::cout << "Display elements in String using array-syntax: " << std::endl;
  for (size_t i = 0; i < str1String.length(); ++i)
  {
    std::cout << "Character [" << i << "] is: " << str1String[i] << std::endl;
  }
  std::cout << "========================" << std::endl;

  // access the contents of a string using iterators.
  std::cout << "Display elements in String using iterators: " << std::endl;
  int charOffset = 0;
  std::string::const_iterator charLocator;
  for (auto charLocator = str1String.cbegin(); charLocator != str1String.cend(); ++charLocator)
  {
    std::cout << "Character [" << charOffset ++ << "] is: " << *charLocator << std::endl;
  }
  std::cout << "========================" << std::endl;

  // access contents as a const char* (C-Style)
  std::cout << "The char* representation of the string is: " << str1String.c_str() << std::endl;
  
  return 0;
}

// $ g++ -std=c++11 -o main 16.2_access_string.cpp 
// $ ./main.exe 

// Display elements in String using array-syntax: 
// Character [0] is: H
// Character [1] is: e
// Character [2] is: l
// Character [3] is: l
// Character [4] is: o
// Character [5] is:
// Character [6] is: S
// Character [7] is: t
// Character [8] is: r
// Character [9] is: i
// Character [10] is: n
// Character [11] is: g
// Character [12] is: .
// ========================
// Display elements in String using iterators:
// Character [0] is: H
// Character [1] is: e
// Character [2] is: l
// Character [3] is: l
// Character [4] is: o
// Character [5] is:
// Character [6] is: S
// Character [7] is: t
// Character [8] is: r
// Character [9] is: i
// Character [10] is: n
// Character [11] is: g
// Character [12] is: .
// ========================
// The char* representation of the string is: Hello String. 