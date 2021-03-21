#include <iostream>
#include <vector>

template <typename T>
void DisplayContainer(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << ' ';
  }
  std::cout << std::endl;
}

// vector<bool> 的成员函数和运算符
int main(int argc, char** argv)
{
  // list initialization for C++11
  std::vector<bool> boolFlags {true, false, true};
  // std::vector<bool> boolFlags(3);
  // boolFlags[0] = true;
  // boolFlags[1] = true;
  // boolFlags[2] = false;

  // insert a fourth bool at the end.
  boolFlags.push_back(true);
  std::cout << "--------------------------------" << std::endl;
  std::cout << "The contents of the vector are: " << std::endl;
  DisplayContainer(boolFlags);

  boolFlags.flip();
  std::cout << "--------------------------------" << std::endl;
  std::cout << "The contents of the vector when flip bits: " << std::endl;
  DisplayContainer(boolFlags);
  
  return 0;
}

// $ g++ -o main 25.4_member_vector_bool.cpp 
// $ ./main.exe 

// --------------------------------
// The contents of the vector are:
// 1 1 0 1
// --------------------------------
// The contents of the vector when flip bits:
// 0 0 1 0