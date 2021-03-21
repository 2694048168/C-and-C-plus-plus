#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

// 使用 for_each() 处理指定范围内的元素
// std::for_each(start of range, end of range, unary function object/unary predicate);

template <typename elementType>
struct DisplayElementKeepcount
{
  int count;
  DisplayElementKeepcount () : count(0) {}

  void operator () (const elementType& element)
  {
    ++count;
    std::cout << element << ' ';
  }
};

int main(int argc, char** argv)
{
  std::vector<int> numInVec {2021, 20, 0, -1, 42, 2020, 25};
  std::cout << "Elements in vector are: " << std::endl;
  DisplayElementKeepcount<int> functor = std::for_each(numInVec.cbegin(), numInVec.cend(), 
                                                      DisplayElementKeepcount<int>());
  std::cout << std::endl;                                                      
  // use the state stored in the return value of for_each.
  std::cout << "'" << functor.count << "' << elements displayed." << std::endl;

  std::string str {"for_each and strings."};
  std::cout << "========================"<< std::endl;
  std::cout << "Sample string: " << str << std::endl;
  
  std::cout << "Characters displayed using lambda: " << std::endl;
  int numChars = 0;
  std::for_each(str.cbegin(), str.cend(), 
                [&numChars] (char c) {std::cout << c << ' '; ++numChars;});
  std::cout << std::endl;
  std::cout << "'" << numChars << "' characters displayed." << std::endl;
  
  return 0;
}

// $ g++ -o main 23.6_for_each_algorithm.cpp 
// $ ./main.exe

// Elements in vector are: 
// 2021 20 0 -1 42 2020 25
// '7' << elements displayed.
// ========================
// Sample string: for_each and strings.
// Characters displayed using lambda:
// f o r _ e a c h   a n d   s t r i n g s .
// '21' characters displayed.