#include <iostream>
#include <string>
#include <cstring>

int main(int argc, char* argv[])
{
  std::cout << "Enter a line of text: " << std::endl;
  std::string userInput;
  getline(std::cin, userInput);

  char copyInput[20] = {'\0'};
  // check string bounds
  if (userInput.length() < 20)
  {
    strcpy(copyInput, userInput.c_str());
    std::cout << "CopyInput contains: " << copyInput << std::endl;
  }
  else
  {
    std::cout << "Bounds exceeded: wont't copy!" << std::endl;
  }
  
  return 0;
}

// $ g++ 6.2_copy_char_string.cpp -o main
// $ ./main.exe 
// Enter a line of text: 
// This fits buffer!
// CopyInput contains: This fits buffer!

// $ ./main.exe 
// Enter a line of text: 
// this does not fit the buffer!
// Bounds exceeded: wont't copy!