#include <iostream>
#include <algorithm>
#include <vector>

// 一元谓词，返回值类型为布尔类型的一元函数称之为一元谓词
// 这种特殊的函数对象用于 STL 算法的判断
// std::partition() 使用一元谓词划分范围
// std::stable_partition() 使用一元谓词划分范围，并保持元素相对位置顺序不变
// std::find_if() 查找函数，根据一元谓词的条件进行判断
// std::remove_if() 删除函数，根据一元谓词的条件进行判断

// a structure as a unary predicate. 
template <typename numberType>
struct IsMultiple
{
  numberType Divisor;

  IsMultiple (const numberType& divisor)
  {
    Divisor = divisor;
  }

  // unary predicate
  bool operator () (const numberType& element) const
  {
    // check if the divisor is a multiple of the divisor.
    return ((element % Divisor) == 0);
  }
};

int main(int argc, char** argv)
{
  std::vector<int> numInVec {42, 24, 66, 99, 4, 1};
  std::cout << "The vector contains: 42, 24, 66, 99, 4, 1" << std::endl;
  std::cout << "Please enter divisor (> 0 ): ";
  int divisor = 2;
  std::cin >> divisor;

  // find the first element that is a multiple of divisor.
  auto element = std::find_if(numInVec.begin(), numInVec.end(), IsMultiple<int>(divisor));

  if (element != numInVec.end())
  {
    std::cout << "First element in vector divisible by " << divisor << " : " << *element << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 21.3_unary_predicate.cpp 
// $ ./main.exe 
// The vector contains: 42, 24, 66, 99, 4, 1
// Please enter divisor (> 0 ): 99
// First element in vector divisible by 99 : 99

// $ ./main.exe 
// The vector contains: 42, 24, 66, 99, 4, 1
// Please enter divisor (> 0 ): 2
// First element in vector divisible by 2 : 42