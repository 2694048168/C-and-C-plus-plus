#include <iostream>
#include <string>

// 拼接字符串
// 1. 使用 复合运算符 += 
// 2. 使用 成员函数 append()

int main(int argc, char** argv)
{
  std::string sampleStr1 ("Hello");
  std::string sampleStr2 (" String.");

  // concatenate
  sampleStr1 += sampleStr2;
  std::cout << "The concatenate of string: " << sampleStr1 << std::endl;

  std::string sampleStr3 (" Fun is not needing to use pointers.");
  sampleStr1.append(sampleStr3);
  std::cout << sampleStr1 << std::endl;

  const char* constCStyleString = " You however still can.";
  sampleStr1.append(constCStyleString);
  std::cout << sampleStr1 << std::endl;
  
  return 0;
}

// $ g++ -std=c++11 -o main 16.3_concat_string.cpp 
// $ ./main.exe 
// The concatenate of string: Hello String.
// Hello String. Fun is not needing to use pointers.
// Hello String. Fun is not needing to use pointers. You however still can.