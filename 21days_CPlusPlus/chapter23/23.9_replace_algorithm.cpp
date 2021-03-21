#include <iostream>
#include <algorithm>
#include <vector>

// 替换值以及替换满足给定条件的元素
// std::replace();
// std::replace_if();

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

int main(int argc, char **argv)
{
  std::vector<int> numInVec(6);

  // fill first 3 elements with value 8, last 3 with 5.
  std::fill(numInVec.begin(), numInVec.begin() + 3, 8);
  std::fill_n(numInVec.begin() + 3, 3, 5);

  // shuffle the container
  /*
  generator by default(1) 
  template <class RandomAccessIterator>
  void random_shuffle(RandomAccessIterator first, RandomAccessIterator last);
  specific generator(2) 
  template <class RandomAccessIterator, class RandomNumberGenerator>
  void random_shuffle(RandomAccessIterator first, RandomAccessIterator last,
                      RandomNumberGenerator && gen);
  */
  std::random_shuffle(numInVec.begin(), numInVec.end());

  std::cout << "The initial contents of vector: " << std::endl;
  DisplayContainer(numInVec);

  std::cout << "==================================" << std::endl;
  std::cout << " 'std::replace' value 5 by 8" << std::endl;
  std::replace(numInVec.begin(), numInVec.end(), 5, 8);
  DisplayContainer(numInVec);

  std::cout << "==================================" << std::endl;
  std::cout << " 'std::replace_if' even value by -1" << std::endl;
  std::replace_if(
      numInVec.begin(), numInVec.end(),
      [](int element) { return ((element % 2) == 0); }, -1);
  DisplayContainer(numInVec);

  return 0;
}

// $ g++ -o main 23.9_replace_algorithm.cpp
// $ ./main.exe

// The initial contents of vector:
// 5 8 5 8 8 5
// | Number of elements: 6
// ==================================
//  'std::replace' value 5 by 8
// 8 8 8 8 8 8
// | Number of elements: 6
// ==================================
//  'std::replace_if' even value by -1
// -1 -1 -1 -1 -1 -1
// | Number of elements: 6
