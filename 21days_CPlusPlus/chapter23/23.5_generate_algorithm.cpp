#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
#include <ctime>  // <time.h>

// 使用 std::generate() 将元素设置为运行阶段生成的值
// std::generate(start of range, end of range, generator function);
// std::generate_n(start of range, n, generator function);

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
  // seed random generator using time.
  srand(time(NULL));

  std::vector<int> numInVec (5);
  std::generate(numInVec.begin(), numInVec.end(), rand);
  std::cout << "Elements in the vctor are: " << std::endl;
  DisplayContainer(numInVec);

  std::list<int> numInList (5);
  std::generate_n(numInList.begin(), 3, rand);
  std::cout << "Elements in the list are: " << std::endl;
  DisplayContainer(numInList);
  
  return 0;
}

// $ g++ -o main 23.5_generate_algorithm.cpp 
// $ ./main.exe 

// Elements in the vctor are: 
// 27055 21828 6230 10140 29320 
// Elements in the list are:    
// 24822 15306 15849 0 0

// $ ./main.exe 

// Elements in the vctor are: 
// 27078 31531 207 14744 3218
// Elements in the list are:
// 19541 4325 28126 0 0