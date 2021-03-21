/** STL algorithm
 * 通用函数，操作容器的，解决一些常用的基本算法，头文件 <algorithm>
 * STL Algorithm: 非变序算法； 便序算法
 * 不改变容器中的元素的顺序和内容的算法，非变序算法；
 * 改变其操作的的序列的元素顺序或内容，变序算法；
 * 主要功能包括：计数，搜索，查找，复制，删除，排序，分区，插入，比较，修改，替换，初始化
 */

#include <iostream>
#include <algorithm>
#include <vector>

// 根据值或者条件查找元素
// iterator std::find(start of range, end of range, element to find); 
// iterator std::find_if(start of range, end of range, unary_predicate);
int main(int argc, char** argv)
{
  std::vector<int> numInVec {2021, 44, -2, 42, 2020, 25};
  std::cout << "Please enter number of find in collection: ";
  int numToFind = 0;
  std::cin >> numToFind;

  auto element = std::find(numInVec.cbegin(), numInVec.cend(), numToFind);
  // check if find succeeded.
  if (element != numInVec.cend())
  {
    // std::cout << "Value " << *element << " found." << std::endl;
    std::cout << "Value " << *element << " found at position [" 
              << std::distance(numInVec.cbegin(), element) << "]" << std::endl;
  }
  else
  {
    std::cout << "No element contains value " << numToFind << std::endl; 
  }

  std::cout << "===========================================" << std::endl;
  std::cout << "Finding the first even number using find_if: " << std::endl;
  
  auto evenNum = std::find_if(numInVec.cbegin(), numInVec.cend(), 
                              [] (int element) {return (element % 2) == 0;});
  // check if find succeeded.
  if (evenNum != numInVec.cend())
  {
    std::cout << "Value " << *evenNum << " found at position [" 
              << std::distance(numInVec.cbegin(), evenNum) << "]" << std::endl;
  }
  else
  {
    std::cout << "No element is even in the collection." << std::endl; 
  }                           
  
  return 0;
}

// $ g++ -o main 23.1_algorithm.cpp 
// $ ./main.exe

// Please enter number of find in collection: 42
// Value 42 found at position [3]
// ===========================================
// Finding the first even number using find_if:
// Value 44 found at position [1]
