#include <iostream>
#include <vector>
#include <algorithm>

template <typename elementType>
class SortAscending
{
public:
  bool operator () (const elementType& num1, const elementType& num2)
  {
    return (num1 < num2);
  }
};

template <typename elementType>
void DisplayContainer(elementType& container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    std::cout << (*element) << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::vector<int> numInVec;
  // insert sample integers
  for (size_t i = 10; i > 0; --i)
  {
    numInVec.push_back(i * 10);
  }

  std::cout << "==============================" << std::endl;
  std::cout << "The origin integers: " << std::endl;
  DisplayContainer(numInVec);

  std::sort(numInVec.begin(), numInVec.end(), SortAscending<int>());
  std::cout << "==============================" << std::endl;
  std::cout << "The after sorting: " << std::endl;
  DisplayContainer(numInVec);
  std::cout << "==============================" << std::endl;

  std::sort(numInVec.begin(), numInVec.end());
  std::cout << "The after sorting using std::less<> : " << std::endl;
  DisplayContainer(numInVec);
  std::cout << "==============================" << std::endl;
  
  return 0;
}

// $ g++ -o main 21.7_test_binary_predicate.cpp 
// $ ./main.exe

// ==============================
// The origin integers:
// 100 90 80 70 60 50 40 30 20 10
// ==============================
// The after sorting:
// 10 20 30 40 50 60 70 80 90 100
// ==============================
// The after sorting using std::less<> :
// 10 20 30 40 50 60 70 80 90 100
// ==============================