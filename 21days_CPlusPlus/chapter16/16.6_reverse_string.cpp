#include <iostream>
#include <string>
#include <algorithm>

// 字符串反转
// 利用泛型算法 std::reverse()

int main(int argc, char** argv)
{
  std::string sampleStr ("Hello string. we will reverse you.");
  std::cout << "The original sample string is: " << std::endl
            << sampleStr << std::endl << std::endl;

  std::reverse(sampleStr.begin(), sampleStr.end());
  
  std::cout << "After applying the std::reverse algorithm: " << std::endl;
  std::cout << sampleStr << std::endl;
  
  return 0;
}

// $ g++ -o main -std=c++17 16.6_reverse_string.cpp 
// $ ./main.exe 

// The original sample string is: 
// Hello string. we will reverse you.

// After applying the std::reverse algorithm:
// .uoy esrever lliw ew .gnirts olleH