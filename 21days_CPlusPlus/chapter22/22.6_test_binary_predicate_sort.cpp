#include <iostream>
#include <algorithm>
#include <vector>

template <typename T>
void DisplayContainer(const T &container)
{
  for (auto element = container.begin(); element != container.end(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::vector<int> vecNumbers {25, -5, 122, 2021, 2020};
  DisplayContainer(vecNumbers);

  std::sort(vecNumbers.begin(), vecNumbers.end());
  DisplayContainer(vecNumbers);

  std::sort(vecNumbers.begin(), vecNumbers.end(),
            [] (int num1, int num2) { return num1 > num2;});
  DisplayContainer(vecNumbers);            

  std::cout << "================================================" << std::endl;
  std::cout << "Number you wish to add to all elements: ";
  int numcontainer = 0;
  std::cin >> numcontainer;
  std::for_each(vecNumbers.begin(), vecNumbers.end(),
                [=] (int& element) {element += numcontainer;});
  DisplayContainer(vecNumbers);                
  
  return 0;
}

// $ g++ -o main 22.6_test_binary_predicate_sort.cpp 
// $ ./main.exe

// 25 -5 122 2021 2020 
// -5 25 122 2020 2021
// 2021 2020 122 25 -5
// ================================================
// Number you wish to add to all elements: 5
// 2026 2025 127 30 0 
