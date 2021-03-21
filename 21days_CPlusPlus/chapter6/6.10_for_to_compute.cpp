#include <iostream>

int main(int argc, char** argv)
{
  // without loop expression(third expression missing)
  for (char userSelection = 'm'; (userSelection != 'x'); )
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
  
  // std::endl 除了换行之外，还能刷新缓冲区
  std::cout << "Goodbye!" << std::endl;
  
  return 0;
}