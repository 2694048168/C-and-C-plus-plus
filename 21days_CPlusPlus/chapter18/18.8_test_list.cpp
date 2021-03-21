#include <iostream>
#include <list>

template <typename T>
void DisplayAsContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

char DisplayOptions()
{
  std::cout << "==========================================" << std::endl;
  std::cout << "What would you like to do?" << std::endl;
  std::cout << "Select 1: To enter an integer." << std::endl;
  std::cout << "Select 2: To display the list." << std::endl;
  std::cout << "Select 0: To quit!" << std::endl;
  std::cout << "==========================================" << std::endl;

  char ch;
  std::cin >> ch;
  return ch;
}

int main(int argc, char** argv)
{
  std::list<int> listData;
  char userSelect = '\0';
  while ((userSelect = DisplayOptions() ) != '0')
  {
    if (userSelect == '1')
    {
      std::cout << "Please enter an integer to be inserted: ";
      int dataInput;
      std::cin >> dataInput;
      listData.push_front(dataInput);
    }
    else if (userSelect == '2')
    {
      std::cout << "Contents of list: " << std::endl;
      DisplayAsContents(listData);
    }
  }
  
  return 0;
}

// $ touch 18.8_test_list.cpp
// $ g++ -o main 18.8_test_list.cpp 
// $ ./main.exe 

// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: To display the list.
// Select 0: To quit!
// ==========================================
// 1
// Please enter an integer to be inserted: 23
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: To display the list.
// Select 0: To quit!
// ==========================================
// 1
// Please enter an integer to be inserted: 42
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: To display the list.
// Select 0: To quit!
// ==========================================
// 1 
// Please enter an integer to be inserted: 2021
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: To display the list.
// Select 0: To quit!
// ==========================================
// 2
// Contents of list: 
// 2021 42 23
// ==========================================
// What would you like to do?
// Select 1: To enter an integer.
// Select 2: To display the list.
// Select 0: To quit!
// ==========================================
// 0