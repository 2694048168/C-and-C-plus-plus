#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  int someNums[] =  {1, 202, -1, 23, 30};
  double moreNums[] = {3.14, 3.141, 3.1415, 3.14159, 3.1415926};
  char charArray[] = {'h', 'e', 'l', 'l', 'o'};
  std::string sayHello = "hello world.";

  for (const int& element : someNums)
  {
    std::cout << element << ' ';
  }
  std::cout << std::endl;

  for (auto element : someNums)
  {
    std::cout << element << ' ';
  }
  std::cout << std::endl;

  for (auto element : charArray)
  {
    std::cout << element << ' ';
  }
  std::cout << std::endl;

  for (auto element : sayHello)
  {
    std::cout << element << ' ';
  }
  std::cout << std::endl;

  for (auto element : moreNums)
  {
    std::cout << element << ' ';
  }
  std::cout << std::endl;

  // range for and auto best using!!!
  // for (auto element : any_type_elements)
  // 遍历任何类型元素的，简洁性高

  return 0;
}

// $ g++ -o main 6.11_range_for.cpp 
// $ ./main.exe 
// 1 202 -1 23 30
// 1 202 -1 23 30
// h e l l o
// h e l l o   w o r l d .
// 3.14 3.141 3.1415 3.14159 3.14159 