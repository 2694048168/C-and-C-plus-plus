#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  // CPlusPlus style string
  // 字符串初始化
  std::string gresstString ("Hello std::string!");
  std::cout << gresstString << std::endl;

  // 储存用户输入的
  std::cout << "Enter a line of text: " << std::endl;
  std::string firstLins;
  getline(std::cin, firstLins);

  std::cout << "Enter another: " << std::endl;
  std::string secondLins;
  getline(std::cin, secondLins);

  // 字符串拼接
  std::cout << "Result of concatenation string: " << std::endl;
  std::string concatString = firstLins + " " + secondLins;
  std::cout << concatString << std::endl;

  // 复制字符串
  std::cout << "Copy of cancatenated string: " << std::endl;
  std::string aCopy;
  aCopy = concatString;
  std::cout << aCopy << std::endl;

  // 确定字符串的长度
  std::cout << "Length of concat string: " << concatString.length() << std::endl;

  return 0;
}

// $ g++ -o main 4.5_CPP_string.cpp 
// $ ./main
// Hello std::string!
// Enter a line of text:
// weili  weili
// Enter another: 
// liwei liwei
// Result of concatenation string: 
// weili  weili liwei liwei
// Copy of cancatenated string:
// weili  weili liwei liwei
// Length of concat string: 24