#include <iostream>
#include <algorithm>
#include <string>

// 字符串的大小写转换
// 利用算法 std::transform()
// 对集合中的每一个元素指向一个用户指定的函数

int main(int argc, char** argv)
{
  std::cout << "Please enter a string for case-convertion:" << std::endl;
  std::cout << ">>";

  std::string inStr;
  getline(std::cin, inStr);
  std::cout << std::endl;

  std::transform(inStr.begin(), inStr.end(), inStr.begin(), ::toupper);
  std::cout << "The string converted to upper case is: " 
            << std::endl << inStr << std::endl << std::endl;
  
  return 0;
}

// $ g++ -o main -std=c++17 16.7_transform_string.cpp 
// $ ./main.exe 

// Please enter a string for case-convertion:
// >>weili

// The string converted to upper case is: 
// WEILI