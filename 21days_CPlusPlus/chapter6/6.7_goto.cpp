#include <iostream>

int main(int argc, char** argv)
{
  // goto 标号
Start:
  int numOne = 0, numTwo = 0; 

  std::cout << "Enter two integers: " << std::endl;
  std::cin >> numOne;
  std::cin >> numTwo;

  std::cout << numOne << " x " << numTwo << " = " << numOne * numTwo << std::endl;
  std::cout << numOne << " + " << numTwo << " = " << numOne + numTwo << std::endl;

  std::cout << "Do you wish to perform another operation (y/n)?" << std::endl;
  char repeat = 'y';
  std::cin >> repeat;

  if (repeat == 'y' || repeat == 'Y')
  {
    goto Start;
  }

  std::cout << "Goodbye!" << std::endl;

  return 0;
}

// $ g++ 6.7_goto.cpp -o main.exe 
// $ ./main.exe 
// Enter two integers: 
// 56
// 25
// 56 x 25 = 1400
// 56 + 25 = 81
// Do you wish to perform another operation (y/n)?
// y
// Enter two integers: 
// 95
// -46
// 95 x -46 = -4370
// 95 + -46 = 49
// Do you wish to perform another operation (y/n)?
// n
// Goodbye!