#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  std::string str1 ("I");
  std::string str2 ("Love");
  std::string str3 ("STL");
  std::string str4 ("string");
  std::string strConcat (str1 + " " + str2 + " " + str3 + " " + str4);

  std::cout << strConcat << std::endl;
  
  return 0;
}

// $ g++ -o main -std=c++14 16.12_test_operators.cpp 
// $ ./main.exe 
// I Love STL string