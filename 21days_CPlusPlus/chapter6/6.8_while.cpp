#include <iostream>

int main(int argc, char** argv)
{
  char userSelection = 'm';

  while (userSelection != 'x')
  {
    std::cout << "Enter the two integers: " << std::endl;
    int numOne = 0, numTwo = 0;
    std::cin >> numOne;
    std::cin >> numTwo;

    std::cout << numOne << " * " << numTwo << " = " << numOne * numTwo << std::endl;
    std::cout << numOne << " + " << numTwo << " = " << numOne + numTwo << std::endl;

    std::cout << "Press x to exit or any other key to recalculate." << std::endl;
    std::cin >> userSelection;
  }

  do
  {
    std::cout << "Enter the two integers: " << std::endl;
    int numOne = 0, numTwo = 0;
    std::cin >> numOne;
    std::cin >> numTwo;

    std::cout << numOne << " * " << numTwo << " = " << numOne * numTwo << std::endl;
    std::cout << numOne << " + " << numTwo << " = " << numOne + numTwo << std::endl;

    std::cout << "Press x to exit or any other key to recalculate." << std::endl;
    std::cin >> userSelection;
  } while (userSelection != 'x');

  
  std::cout << "Goodbye!" << std::endl;

  // while 与 do-while 区别在于后者必须执行一次，而前者可能一次都不执行
  
  return 0;
}