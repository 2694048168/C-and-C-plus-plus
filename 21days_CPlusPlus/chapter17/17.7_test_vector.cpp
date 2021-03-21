#include <iostream>
#include <vector>
#include <algorithm>

char DisplayOptions()
{
  std::cout << "==========================================" << std::endl;
  std::cout << "What would you like to do?" << std::endl;
  std::cout << "Select 1: To enter an integer." << std::endl;
  std::cout << "Select 2: Query a value given an index." << std::endl;
  std::cout << "Select 3: Query a value." << std::endl;
  std::cout << "Select 4: To display the vector." << std::endl;
  std::cout << "Select 0: To quit!" << std::endl;
  std::cout << "==========================================" << std::endl;

  char ch;
  std::cin >> ch;
  return ch;
}

int main(int argc, char** argv)
{
  std::vector<int> vecData;
  char userSelect = '\0';
  while ((userSelect = DisplayOptions() ) != '0')
  {
    if (userSelect == '1')
    {
      std::cout << "Please enter an integer to be inserted: ";
      int dataInput;
      std::cin >> dataInput;
      vecData.push_back(dataInput);
    }
    else if (userSelect == '2')
    {
      std::cout << "Please enter an index betweet 0 and " << (vecData.size() -1) << ": ";
      size_t index = 0;
      std::cin >> index;

      if (index < (vecData.size()))
      {
        std::cout << "Element [" << index << "] = " << vecData[index] << std::endl;
      }
    }
    else if (userSelect == '3')
    {
      // test std::find()
      std::cout << "Please enter the value that you want to find: ";
      int value;
      std::cin >> value;
      std::vector<int>::const_iterator elementFound = std::find(vecData.begin(), vecData.end(), value);
      if (elementFound != vecData.end())
      {
        std::cout << "Element found in the vector." << std::endl;
      }
      else
      {
        std::cout << "Element not found int the vector." << std::endl;
      }
    }
    else if (userSelect == '4')
    {
      std::cout << "The contents of the vector are: ";
      for (size_t i = 0; i < vecData.size(); ++i)
      {
        std::cout << vecData[i] << ' ';
      }
      std::cout << std::endl;
    }
  }
  
  return 0;
}

// $ g++ -o main 17.7_test_vector.cpp                             
// $ ./main.exe 

// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 1
// Please enter an integer to be inserted: 42
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 1
// Please enter an integer to be inserted: 24
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 1
// Please enter an integer to be inserted: 2021
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 2
// Please enter an index betweet 0 and 2: 2
// Element [2] = 2021
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 3
// Please enter the value that you want to find: 42
// Element found in the vector.
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 3
// Please enter the value that you want to find: 2020
// Element not found int the vector.
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 4
// The contents of the vector are: 42 24 2021 
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: Query a value given an index.
// Select 3: Query a value.
// Select 4: To display the vector.
// Select 0: To quit!
// ==========================================
// 0
