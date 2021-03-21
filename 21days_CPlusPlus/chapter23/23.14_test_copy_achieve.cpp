#include <iostream>
#include <algorithm>
#include <list>
#include <string>
#include <vector>

template <typename T>
void DisplayContainer(const T &container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char **argv)
{
  std::list<std::string> listNames;
  listNames.push_back("Jack");
  listNames.push_back("John");
  listNames.push_back("Anna");
  listNames.push_back("Skate");

  std::list<std::string>::const_iterator iListNames;
  for (iListNames = listNames.begin(); iListNames != listNames.end(); ++iListNames)
  {
    std::cout << *iListNames << ' ';
  }

  std::cout << std::endl << "-----------------------" << std::endl;
  std::vector<std::string> vecNames(4);
  std::copy(listNames.begin(), listNames.end(), vecNames.begin());

  std::vector<std::string>::const_iterator iNames;
  for (iNames = vecNames.begin(); iNames != vecNames.end(); ++iNames)
  {
    std::cout << *iNames << ' ';
  }

  return 0;
}

// $ g++ -o main 23.14_test_copy_achieve.cpp
// $ ./main.exe

// Jack John Anna Skate 
// -----------------------
// Jack John Anna Skate
