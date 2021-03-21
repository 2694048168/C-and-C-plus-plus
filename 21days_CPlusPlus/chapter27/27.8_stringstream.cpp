#include <iostream>
#include <fstream>
#include <sstream>

int main(int argc, char** argv)
{
  std::cout << "Please enter an integer: ";
  int input = 0;
  std::cin >> input;

  std::stringstream converterStream;
  converterStream << input;
  std::string inputAsStr;
  converterStream >> inputAsStr;

  std::cout << "Integer input = " << input << std::endl;
  std::cout << "String gained from integer = " << inputAsStr << std::endl;

  std::stringstream anotherStream;
  anotherStream << inputAsStr;
  int copy = 0;
  anotherStream >> copy;

  std::cout << "Integer gained from string, copy = " << copy << std::endl;

  return 0;
}

// $ g++ -o main 27.8_stringstream.cpp 
// $ ./main.exe 

// Please enter an integer: 42
// Integer input = 42
// String gained from integer = 42      
// Integer gained from string, copy = 42