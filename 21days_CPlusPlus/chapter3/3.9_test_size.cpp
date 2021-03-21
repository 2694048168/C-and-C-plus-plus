#include <iostream>

int main(int argc, char**argv)
{
  std::cout << "size of int : " << sizeof(int) << std::endl;
  std::cout << "size of unsigned int : " << sizeof(unsigned int) << std::endl;
  std::cout << "**********************" << std::endl;
  std::cout << "size of unsigned long long : " << sizeof(unsigned long long) << std::endl;
  std::cout << "size of long long : " << sizeof(long long) << std::endl;
  std::cout << "size of long : " << sizeof(long) << std::endl;
  

  std::cout << "The output changes with compiler, hardware and OS." << std::endl;

  return 0;
}

// $ g++ -o main 3.9_test_size.cpp
// $ ./main
// size of int : 4
// size of unsigned int : 4
// **********************
// size of unsigned long long : 8
// size of long long : 8
// size of long : 4
// The output changes with compiler, hardware and OS.