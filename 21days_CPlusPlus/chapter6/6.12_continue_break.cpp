#include <iostream>

int main(int argc, char** argv)
{
  // 无限循环
  // 使用 continue 进入直接下一次循环
  // 使用 break 跳出无限循环
  for (; ;)
  {
    std::cout << "Enter the two integers: " << std::endl;
    int numOne = 0, numTwo = 0;
    std::cin >> numOne;
    std::cin >> numTwo;

    std::cout << "Do you wish to correct the numbers? (y/n): ";
    char changeNumbers = '\0';
    std::cin >> changeNumbers;

    if (changeNumbers == 'y' || changeNumbers == 'Y')
    {
      // continue upto restart the loop
      continue;
    }
    
    std::cout << numOne << " * " << numTwo << " = " << numOne * numTwo << std::endl;
    std::cout << numOne << " + " << numTwo << " = " << numOne + numTwo << std::endl;

    std::cout << "Press x to exit or any other key to recalculate." << std::endl;
    char userSelection = '\0';
    std::cin >> userSelection;

    if (userSelection == 'x' || userSelection == 'X')
    {
      // break upto exit the loop
      break;
    }
  }
    
  std::cout << "Goodbye!" << std::endl;

  return 0;
}