/**标准模板库 Standard Template Library, STL
 * 1. 储存数据的容器 container
 * 2. 访问数据的迭代器 iterator
 * 3. 操作数据的算法 algorithm
 * 
 * STL container: 
 * 1. 顺序容器: std::vector, std::deque, std::list, std::forward_list
 * 2. 关联容器: std::set, std::unordered_set, std::multiset, std::unordered_multiset;
 * 2. 关联容器: std::map, std::unordered_map, std::multimap, std::unordered_multimap;
 * 3. 容器适配器 container adapter: std::stack(LIFO), std::queue(FIFO), std::priority_queue
 * 
 * STL iterator
 * 1. 输入迭代器
 * 2. 输出迭代器
 * 3. 前向迭代器
 * 4. 双向迭代器
 * 5. 随机访问迭代器
 * 
 * STL algorithm
 * 0. #include <algorithm>
 * 1. 查找: std::find, std::find_if;
 * 2. 反转: std::reverse, std::reverse_if;
 * 3. 变换: std::transform;
 * 
 * 迭代器就是容器和算法之间的桥梁
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include <deque>

int main(int argc, char** argv)
{
  // 将下面的所有 vector 全部替换为 deque , 代码能够编译通过并运行;
  // 这表明迭代器可以帮助更高简单高效的使用算法和容器.

  // A dynamic array of integers
  // std::vector<int> intArray;
  std::deque<int> intArray;

  // insert sample integers into the array
  intArray.push_back(50);
  intArray.push_back(42);
  intArray.push_back(21);
  intArray.push_back(2021);
  intArray.push_back(2020);

  // std::cout << "The contents of the vector are: " << std::endl;
  std::cout << "The contents of the deque are: " << std::endl;

  // walk the vector and read values using an iterator
  // std::vector<int>::iterator arrIterator = intArray.begin();
  // walk the deque and read values using an iterator
  std::deque<int>::iterator arrIterator = intArray.begin();

  while (arrIterator != intArray.end())
  {
    // write the value to the screen
    std::cout << *arrIterator << std::endl;

    // increment the iterator to access the next element
    ++arrIterator;
  }

  // find an element(2021) using the 'find' algorithm
  // std::vector<int>::iterator eleFound = std::find(intArray.begin(), intArray.end(), 2021);
  std::deque<int>::iterator eleFound = std::find(intArray.begin(), intArray.end(), 2021);

  // check if value was found
  if (eleFound != intArray.end())
  {
    // determine position of element using std::distance
    int elePos = std::distance(intArray.begin(), eleFound);
    std::cout << "Value " << *eleFound << " found in the vector at position: " << elePos << std::endl;
  }
  
  return 0;
}

// $ g++ -std=c++11 -o main 15.1_vector_find_itera.cpp 
// $ g++ -std=c++14 -o main 15.1_vector_find_itera.cpp 
// $ g++ -std=c++17 -o main 15.1_vector_find_itera.cpp 
// $ g++ -std=c++2a -o main 15.1_vector_find_itera.cpp 
// $ ./main.exe 
// The contents of the vector are: 
// 50
// 42
// 21
// 2021
// 2020
// Value 2021 found in the vector at position: 3

// $ g++ -std=c++11 -o main 15.1_vector_find_itera.cpp 
// $ ./main.exe 
// The contents of the deque are: 
// 50
// 42
// 21
// 2021
// 2020
// Value 2021 found in the vector at position: 