#include <iostream>

constexpr int Square(int number) { return number * number; }

int main(int argc, char**argv)
{
  int arrayNumbers [5] = {34, 56, -21, 5002, 365};

  // std::cout << "对数组元素进行访问: " << std::endl;
  std::cout << "First element at index 0: " << arrayNumbers[0] << std::endl;
  std::cout << "First element at index 1: " << arrayNumbers[1] << std::endl;
  std::cout << "First element at index 2: " << arrayNumbers[2] << std::endl;
  std::cout << "First element at index 3: " << arrayNumbers[3] << std::endl;
  std::cout << "First element at index 4: " << arrayNumbers[4] << std::endl;

  const int ARRAY_LENGTH = 5;

  // Array of 5 integers, initialized using a const
  int lessNumbers [ARRAY_LENGTH] = {5, 10, 0, -101, 20};

  // Using a constexpr for array of 25 integers
  int moreNumbers [ARRAY_LENGTH] = {Square(ARRAY_LENGTH)};

  std::cout << "=================" << std::endl;
  std::cout << "Enter index of the element to be changed: ";
  int elementIndex = 0;
  std::cin >> elementIndex;

  std::cout << "Enter new value: ";
  int newValue = 0;
  std::cin >> newValue;

  lessNumbers[elementIndex] = newValue;
  moreNumbers[elementIndex] = newValue;

  std::cout << "Element " << elementIndex << " in array lessNumbers is: "
            << lessNumbers[elementIndex] << std::endl;
  std::cout << "Element " << elementIndex << " in array moreNumbers is: "
            << moreNumbers[elementIndex] << std::endl;

  return 0;
}

// $ g++ -o main 4.1_array.cpp 
// $ ./main.exe 
// First element at index 0: 34 
// First element at index 1: 56 
// First element at index 2: -21
// First element at index 3: 5002
// First element at index 4: 365
// =================
// Enter index of the element to be changed: 3
// Enter new value: 111
// Element 3 in array lessNumbers is: 111
// Element 3 in array moreNumbers is: 111
