#include <iostream>
#include <list>
#include <vector>

template <typename T>
void DisplayAsContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::list<int> listData {1, 5, 3, 7, 9};
  std::vector<int> vectorData {2, 4, 0, 8, 6};

  std::cout << "The list: ";
  DisplayAsContents(listData);

  std::cout << "The vector: ";
  DisplayAsContents(vectorData);

  listData.insert(listData.end(), vectorData.begin(), vectorData.end());
  std::cout << "After insert: ";
  DisplayAsContents(listData);
  
  return 0;
}

// $ g++ -o main 18.10_test_container.cpp 
// $ ./main.exe 

// The list: 1 5 3 7 9 
// The vector: 2 4 0 8 6
// After insert: 1 5 3 7 9 2 4 0 8 6