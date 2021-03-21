#include <iostream>
#include <string>
#include <algorithm>

int main(int argc, char** argv)
{
  std::cout << "Please enter a string for case-conversion:" << std::endl << ">>";
  std::string strInput;
  getline(std::cin, strInput);
  std::cout << std::endl;

  for (size_t i = 0; i < strInput.length(); i += 2)
  {
    strInput[i] = toupper(strInput[i]);
  }

  std::cout << "The string converted to upper cass is:" << std::endl;
  std::cout << strInput << std::endl << std::endl;
  
  return 0;
}

// $ g++ -o main 16.11_test_upper.cpp 
// $ ./main.exe 
// Please enter a string for case-conversion:
// >>weili

// The string converted to upper cass is:
// WeIlI