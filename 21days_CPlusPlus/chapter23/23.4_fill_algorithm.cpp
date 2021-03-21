#include <iostream>
#include <algorithm>
#include <vector>

// 将容器中的元素初始化为指定值
// std::fill(start of fill, end of fill, fill value);
// std::fill_n(start of fill, n, fill value);

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
  // initialize a sample vector with 3 element.
  std::vector<int> numInVec (3);

  // fill all element in the container with value 9.
  std::fill(numInVec.begin(), numInVec.end(), 9);

  // increase the size of the vector to hold 6 element.
  // 一次性申请内存，避免频繁申请内存
  numInVec.resize(6);

  // fill the three elements starting at offset posistion 3 with value -9.
  // 除非必要，尽量使用常量迭代器
  std::fill_n(numInVec.begin() + 3, 3, -9);

  std::cout << "Contents of the vector are: " << std::endl;
  DisplayContainer(numInVec);  
  
  return 0;
}

// $ g++ -o main 23.4_fill_algorithm.cpp 
// $ ./main.exe 

// Contents of the vector are: 
// 9 9 9 -9 -9 -9