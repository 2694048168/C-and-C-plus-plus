#include <iostream>
#include <algorithm>
#include <vector>

// 将范围分区
// std::partition();
// std::stable_partition();

// unary predicate
bool IsEven (const int& num)
{
  return ((num % 2) == 0);
}

template <typename T>
void DisplayContainer(const T &container)
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
  std::vector<int> numInVec {2021, 2020, 0, -1, 42, 25, 5, 3};
  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "The initial contents: " << std::endl;
  DisplayContainer(numInVec);

  std::vector<int> vecCopy(numInVec);
  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "The effect of using partition() : " << std::endl;
  std::partition(numInVec.begin(), numInVec.end(), IsEven);
  DisplayContainer(numInVec);

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "The effect of using stable_partition() : " << std::endl;
  std::stable_partition(vecCopy.begin(), vecCopy.end(), IsEven);
  DisplayContainer(numInVec);
  
  return 0;
}

// $ g++ -o main 23.11_partition_algorithm.cpp 
// $ ./main.exe

// -----------------------------------------
// The initial contents: 
// 2021 2020 0 -1 42 25 5 3
// | Number of elements: 8
// -----------------------------------------
// The effect of using partition() :
// 42 2020 0 -1 2021 25 5 3
// | Number of elements: 8
// -----------------------------------------
// The effect of using stable_partition() : 
// 42 2020 0 -1 2021 25 5 3
// | Number of elements: 8