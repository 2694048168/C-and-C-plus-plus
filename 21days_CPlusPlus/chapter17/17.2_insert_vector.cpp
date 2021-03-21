#include <iostream>
#include <vector>

// using insert() 在指定位置插入元素
void DisplayVector(const std::vector<int>& inVec)
{
  for (auto element = inVec.cbegin(); element != inVec.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  // instantiate a vector with 4 elements, each initialized to 90.
  std::vector<int> integers (4, 90);

  std::cout << "The initial contents of the vector: ";
  DisplayVector(integers);

  // insert 25 at the beginning.
  integers.insert(integers.begin(), 25);

  // insert 2 numbers of value 45 at the end.
  integers.insert(integers.end(), 2, 45);

  std::cout << "=======================================" << std::endl;
  DisplayVector(integers);

  // another vector containing 2 elements of value 30.
  std::vector<int> another (2, 30);

  // insert two elements from another container in position [1]
  integers.insert(integers.begin() + 1, another.begin(), another.end());

  std::cout << "=======================================" << std::endl;
  std::cout << "Vector after inserting contents from another vector: in the middle:" << std::endl;
  DisplayVector(integers);
  
  return 0;
}

// $ g++ -o main 17.2_insert_vector.cpp 
// $ ./main.exe 
// The initial contents of the vector: 90 90 90 90
// =======================================
// 25 90 90 90 90 45 45
// =======================================
// Vector after inserting contents from another vector: in the middle:
// 25 30 30 90 90 90 90 45 45