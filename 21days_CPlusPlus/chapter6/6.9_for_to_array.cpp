#include <iostream>

int main(int argc, char** argv)
{
  const int ARRAY_LENGTH = 6;
  int numArray[ARRAY_LENGTH] = {0};

  std::cout << "Populate array of " << ARRAY_LENGTH << " integers" << std::endl;

  for (size_t i = 0; i < ARRAY_LENGTH; ++i)
  {
    std::cout << "Enter an integer for element " << i << ": ";
    std::cin >> numArray[i];
  }

  std::cout << "*************************" << std::endl;
  std::cout << "Displaying contents of the array: " << std::endl;

  for (size_t i = 0; i < ARRAY_LENGTH; ++i)
  {
    std::cout << "Element " << i << " = " << numArray[i] << std::endl;
  }
  
  return 0;
}


// $ g++ -o main 6.9_for_to_array.cpp 
// $ ./main.exe
// Populate array of 6 integers    
// Enter an integer for element 0: 1
// Enter an integer for element 1: 2
// Enter an integer for element 2: 3
// Enter an integer for element 3: 4
// Enter an integer for element 4: 6
// Enter an integer for element 5: 8
// *************************
// Displaying contents of the array: 
// Element 0 = 1
// Element 1 = 2
// Element 2 = 3
// Element 3 = 4
// Element 4 = 6
// Element 5 = 8