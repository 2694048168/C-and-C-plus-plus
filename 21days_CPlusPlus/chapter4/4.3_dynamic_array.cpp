#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
  // dynamic array of integer
  std::vector<int> dynamicArray (3);

  dynamicArray[0] = 365;
  dynamicArray[1] = -421;
  dynamicArray[2] = 789;

  std::cout << "Number of integers in array: " << dynamicArray.size() << std::endl;

  std::cout << "Enter another element to insert ";
  int newValue = 0;
  std::cin >> newValue;
  dynamicArray.push_back(newValue);

  std::cout << "Number of integers in array: " << dynamicArray.size() << std::endl;
  std::cout << "Last element in array: " << dynamicArray[dynamicArray.size() - 1] << std::endl;

  return 0;
}

// $ g++ -o main 4.3_dynamic_array.cpp 
// $ ./main.exe
// Number of integers in array: 3  
// Enter another element to insert 35
// Number of integers in array: 4
// Last element in array: 35