#include <iostream>

int main(int argc, char**argv)
{
  std::cout << "size of bool: " << sizeof(bool) << std::endl;
  std::cout << "size of char: " << sizeof(char) << std::endl;
  std::cout << "size of unsigned short int : " << sizeof(unsigned short) << std::endl;
  std::cout << "size of short int : " << sizeof(short) << std::endl;
  std::cout << "size of unsigned long int : " << sizeof(unsigned long) << std::endl;
  std::cout << "size of long int : " << sizeof(long) << std::endl;
  std::cout << "size of int : " << sizeof(int) << std::endl;
  std::cout << "size of unsigned long long : " << sizeof(unsigned long long) << std::endl;
  std::cout << "size of long long : " << sizeof(long long) << std::endl;
  std::cout << "size of unsigned int : " << sizeof(unsigned int) << std::endl;
  std::cout << "size of float : " << sizeof(float) << std::endl;
  std::cout << "size of double : " << sizeof(double) << std::endl;

  std::cout << "The output changes with compiler, hardware and OS." << std::endl;

  return 0;
}


// $ g++ -o main 3.5_variable_sizeof.cpp
// $ ./main.exe 
// size of bool: 1
// size of char: 1
// size of unsigned short int : 2
// size of short int : 2
// size of unsigned long int : 4
// size of long int : 4
// size of int : 4
// size of unsigned long long : 8
// size of long long : 8
// size of unsigned int : 4
// size of float : 4
// size of double : 8
// The output changes with compiler, hardware and OS.