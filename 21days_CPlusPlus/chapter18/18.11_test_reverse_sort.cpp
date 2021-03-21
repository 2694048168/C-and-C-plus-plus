#include <iostream>
#include <list>
#include <string>

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
  std::list<std::string> listDataStr;
  listDataStr.push_back("Jack");
  listDataStr.push_back("Skat");
  listDataStr.push_back("Anna");
  listDataStr.push_back("John");

  std::cout << "The list: ";
  DisplayAsContents(listDataStr);

  std::cout << "The list after reverse: ";
  listDataStr.reverse();
  DisplayAsContents(listDataStr);

  std::cout << "The vector after sort: ";
  DisplayAsContents(listDataStr);

  return 0;
}

// $ g++ -o main 18.11_test_reverse_sort.cpp 
// $ ./main.exe

// The list: Jack Skat Anna John 
// The list after reverse: John Anna Skat Jack
// The vector after sort: John Anna Skat Jack