#include <iostream>
#include <algorithm>
#include <vector>
#include <list>

// 复制和删除元素
// 1. std::copy(); std::copy_if(); std::copy_backward(); 
// std::copy(start source range, end source range, start dest range);
// std::copy_if(start source, end source, copy position in dest range, unary predicate);
// std::copy_backward(start source, end source, dist end range);
// 2. std::remove(); std::rmove_if();
// std::remove(start range, end range, remove value);
// std::remove_if(start range, end range, unary predicate);

template <typename T>
void DisplayContainer (const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;
  std::cout << "| Number of elements: " << container.size() << std::endl;
}

int main(int argc, char** argv)
{
  std::list<int> numInList {2021, 0, -2, 42, 2020, 25};
  std::cout << "Source (list) contains:" << std::endl;
  DisplayContainer(numInList);

  // initializez vector to hold 2x elements as the list.
  std::vector<int> numInVec (numInList.size() * 2);
  auto lastElement = std::copy(numInList.cbegin(), numInList.cend(), numInVec.begin());
  // copy add numbers from list into vector.
  std::copy_if(numInList.cbegin(), numInList.cend(), lastElement,
               [] (int element) {return ((element % 2) != 0);});
  std::cout << "Destination (vector) after copy and copy_if:" << std::endl;
  DisplayContainer(numInVec);

  // remove all instances of '0' , resize vector using erase()
  auto newEnd = std::remove(numInVec.begin(), numInVec.end(), 0);
  numInVec.erase(newEnd, numInVec.end());

  // remove all odd numbers from the vector using remove_if.
  newEnd = std::remove_if(numInVec.begin(), numInVec.end(),
                          [] (int element) {return ((element % 2) != 0);});
  numInVec.erase(newEnd, numInVec.end());                          

  std::cout << "Destination (vecto) after remove, remove_if, erase:" << std::endl;
  DisplayContainer(numInVec);

  return 0;
}

// $ g++ -o main 23.8_copy_remove_algorithm.cpp 
// $ ./main.exe 

// Source (list) contains:
// 2021 0 -2 42 2020 25
// | Number of elements: 6
// Destination (vector) after copy and copy_if:
// 2021 0 -2 42 2020 25 2021 25 0 0 0 0
// | Number of elements: 12
// Destination (vecto) after remove, remove_if, erase:
// -2 42 2020
// | Number of elements: 3
