#include <iostream>
#include <map>

template <typename T>
void DisplayContent (const T& container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    // std::cout << element->first << " ---> " << element->second << std::endl;
    std::cout << (*element).first << " ---> " << (*element).second << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::multimap<int, int> mmapIntegers;
  mmapIntegers.insert(std::make_pair(5, 42));
  mmapIntegers.insert(std::make_pair(5, 42));
  mmapIntegers.insert(std::make_pair(5, 42));
  mmapIntegers.insert(std::make_pair(5, 42));
  mmapIntegers.insert(std::make_pair(5, 42));

  std::map<int, int> mapIntegers;
  mapIntegers.insert(std::make_pair(5, 24));
  mapIntegers.insert(std::make_pair(5, 24));
  mapIntegers.insert(std::make_pair(5, 24));
  mapIntegers.insert(std::make_pair(5, 24));
  mapIntegers.insert(std::make_pair(5, 24));

  std::cout << "Displaying the contents of the multimap: " << std::endl;
  DisplayContent(mmapIntegers);
  std::cout << "The size of the multimap is: " << mmapIntegers.size() << std::endl;
  std::cout << "=========================================" << std::endl;

  std::cout << "Displaying the contents of the map: " << std::endl;
  DisplayContent(mapIntegers);
  std::cout << "The size of the map is: " << mapIntegers.size() << std::endl;
  std::cout << "=========================================" << std::endl;
  
  return 0;
}

// $ g++ -o main 20.7_test_repeat_element.cpp 
// $ ./main.exe

// Displaying the contents of the multimap: 
// 5 ---> 42
// 5 ---> 42
// 5 ---> 42
// 5 ---> 42
// 5 ---> 42

// The size of the multimap is: 5
// =========================================
// Displaying the contents of the map:
// 5 ---> 24

// The size of the map is: 1
// =========================================