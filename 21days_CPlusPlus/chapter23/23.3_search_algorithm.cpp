#include <iostream>
#include <algorithm>
#include <vector>
#include <list>

// 在集合中搜索元素或者序列
// auto range = std::search(start range to search in, end range to search in, 
//                          start range to search, end range of search for)
// auto partialRange = std::search_n(start range, end range, num item to be searched for, 
//                                   value to search for)

template <typename T>
void DisplayContainer (const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  std::vector<int> numInVec {2021, 2020, 0, -1, 42, 1010, 25, 9, 9, 9};
  std::list<int> numInList {-1, 42, 1010};

  std::cout << "The contents of the sample vector are: " << std::endl;
  DisplayContainer(numInVec);
  std::cout << "=============================================" << std::endl;
  std::cout << "The contents of the sample list are: " << std::endl;
  DisplayContainer(numInList);

  std::cout << "=============================================" << std::endl;
  std::cout << "search() for the contents fo list in vector: " << std::endl;
  auto range = std::search(numInVec.cbegin(), numInVec.cend(), numInList.cbegin(), numInList.cend());
  if (range != numInVec.cend())
  {
    std::cout << "Sequence in list found in vector at position: " 
              << std::distance(numInVec.cbegin(), range) << std::endl;
  }
  
  std::cout << "=============================================" << std::endl;
  std::cout << "search{9, 9, 9} in vector using search_n(): " << std::endl;
  auto partialRange = std::search_n(numInVec.cbegin(), numInVec.cend(), 3, 9);
  if (partialRange != numInVec.cend())
  {
    std::cout << "Sequence {9, 9, 9} found in vector at position: " 
              << std::distance(numInVec.cbegin(), partialRange) << std::endl;
  }

  return 0;
}

// $ g++ -o main 23.3_search_algorithm.cpp 
// $ ./main.exe 

// The contents of the sample vector are: 
// 2021 2020 0 -1 42 1010 25 9 9 9
// =============================================    
// The contents of the sample list are:
// -1 42 1010
// =============================================    
// search() for the contents fo list in vector:     
// Sequence in list found in vector at position: 3  
// =============================================    
// search{9, 9, 9} in vector using search_n():      
// Sequence {9, 9, 9} found in vector at position: 7