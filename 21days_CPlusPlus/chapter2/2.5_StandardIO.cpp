#include <iostream>
#include <string>

int main(int argc, char**argv)
{
  // Delcare a variable to store an integer
  int inputNumber = 0;

  std::cout << "Enter an integer: ";

  // Store integer given user input
  std::cin >> inputNumber;

  // The same with text string data
  std::cout << "Enter youe name: ";
  std::string inputName = "";
  std::cin >> inputName;

  std::cout << inputName << " entered " << inputNumber << std::endl;

  return 0;
}