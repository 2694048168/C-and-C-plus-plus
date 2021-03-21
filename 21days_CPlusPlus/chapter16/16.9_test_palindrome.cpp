#include <iostream>
#include <string>
#include <algorithm>

int main(int argc, char** argv)
{
  std::cout << "Please enter a word for palindrome-check:" << std::endl;
  std::string strInput;
  std::cin >> strInput;

  // 复制一份副本，反转之后与原始字符串进行对比
  std::string strCopy(strInput);
  std::reverse(strCopy.begin(), strCopy.end());
  if (strCopy == strInput)
  {
    std::cout << strInput << " is a palindrome." << std::endl;
  }
  else
  {
    std::cout << strInput << " is not a palindrome." << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 16.9_test_palindrome.cpp 
// $ ./main.exe 
// Please enter a word for palindrome-check:
// liwei
// liwei is not a palindrome.

// $ ./main.exe 
// Please enter a word for palindrome-check:
// atoota
// atoota is a palindrome.