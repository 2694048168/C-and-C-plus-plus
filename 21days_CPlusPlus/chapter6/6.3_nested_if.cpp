#include <iostream>

int main(int argc, char** argv)
{
  std::cout << "Enter two numbers: " << std::endl;
  float numOne = 0, numTwo = 0;
  std::cin >> numOne;
  std::cin >> numTwo;

  std::cout << "Enter 'd' to divide, anything else to multiply: ";
  char userSelection = '\0';
  std::cin >> userSelection;

  if (userSelection == 'd')
  {
    std::cout << "You wish to divide!" << std::endl;
    if (numTwo)
    {
      std::cout << numOne << " / " << numTwo << " = " << numOne / numTwo << std::endl;
    }
    else
    {
      std::cout << "Division by zero is not allowed." << std::endl;
    }
  }
  else
  {
    std::cout << "You wish to multiply!" << std::endl;
    std::cout << numOne << " * " << numTwo << " = " << numOne * numTwo << std::endl;
  }
  
  return 0 ;
}

// $ g++ 6.3_nested_if.cpp -o main.exe 

// $ ./main.exe
// Enter two numbers: 
// 12
// 0
// Enter 'd' to divide, anything else to multiply: d
// You wish to divide!
// Division by zero is not allowed.

// $ ./main.exe 
// Enter two numbers: 
// 12
// 2
// Enter 'd' to divide, anything else to multiply: d
// You wish to divide!
// 12 / 2 = 6

// $ ./main.exe 
// Enter two numbers: 
// 12
// 2
// Enter 'd' to divide, anything else to multiply: a
// You wish to multiply!
// 12 * 2 = 24