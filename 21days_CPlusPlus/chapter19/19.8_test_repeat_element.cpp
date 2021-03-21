#include <iostream>
#include <set>

template <typename T>
void DisplayContent (const T& container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::multiset<int> msetIntegers;
  msetIntegers.insert(5);
  msetIntegers.insert(5);
  msetIntegers.insert(5);

  std::set<int> setIntegers;
  setIntegers.insert(5);
  setIntegers.insert(5);
  setIntegers.insert(5);

  std::cout << "Displaying the contents of the multiset: ";
  DisplayContent(msetIntegers);
  std::cout << "The size of the multiset is: " << msetIntegers.size() << std::endl;
  std::cout << "=========================================" << std::endl;

  std::cout << "Displaying the contents of the set: ";
  DisplayContent(setIntegers);
  std::cout << "The size of the set is: " << setIntegers.size() << std::endl;
  std::cout << "=========================================" << std::endl;
  
  return 0;
}

// $ g++ -o main 19.8_test_repeat_element.cpp 
// $ ./main.exe 

// Displaying the contents of the multiset: 5 5 5 
// The size of the multiset is: 3
// =========================================
// Displaying the contents of the set: 5
// The size of the set is: 1
// =========================================