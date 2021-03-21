#include <iostream>
#include <vector>  // vector<bool>
#include <string>
#include <bitset>

// std::bitset 缺点就是不能动态的调整长度，必须在编译阶段决定
// 为了克服，提供了 vector<bool> 或者 bit_vector
// 可以动态调整长度
template <typename T>
void DisplayContainer(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  // instantiate an object using the dufault constructor.
  std::vector<bool> boolFlags1;
  std::cout << "----------------------" << std::endl;
  DisplayContainer(boolFlags1);

  // initialize a vector with 10 elements with value true.
  std::vector<bool> boolFlags2(10, true);
  std::cout << "----------------------" << std::endl;
  DisplayContainer(boolFlags2);

  // initialize one object as a copy of another.
  std::vector<bool> boolFlags3(boolFlags2);
  std::cout << "----------------------" << std::endl;
  DisplayContainer(boolFlags3);

  return 0;
}

// $ g++ -o main 25.3_vector_bool.cpp 
// $ ./main.exe

// ----------------------

// ----------------------
// 1 1 1 1 1 1 1 1 1 1
// ----------------------
// 1 1 1 1 1 1 1 1 1 1