#include <iostream>
#include <set>

// STL 集合类
// std::set; std::multiset; std::unordered_set; std::unordered_multiset
// 储存一组经过排序的元素，内部结构类似二叉树，提高查找速度
// 默认使用 std::less 谓词进行排序
// 也可以自定义排序谓词函数，进行参数传入

// used as a template parameter in set / multiset instantialtion
template <typename T>
struct SortDescending
{
  bool operator() (const T& lhs, const T& rhs) const
  {
    return (lhs > rhs);
  }
};

template <typename T>
void DisplayContents(const T& container)
{
  for (auto element = container.cbegin(); element != container.cend(); ++element)
  {
    std::cout << *element << " ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv)
{
  // 1. create set / multiset.
  // a simple set or multiset of integers with default sort predicate.
  // list initialization
  std::set<int> setInt1 {2020, 2021, 42, 24, -1};
  std::multiset<int> msetInt2;

  // set and multiset instantiated given a user-defined sort predicate.
  std::set<int, SortDescending<int>> setInt3; 
  std::multiset<int, SortDescending<int>> msetInt4; 

  // creating one set from another, or part of another container.
  std::set<int> setInt5(setInt1);
  std::multiset<int> msetInt6(setInt1.cbegin(), setInt1.cend());

  // 2. insert elements.
  setInt1.insert(-1);  // duplicate
  std::cout << "Contents of the set: " << std::endl;
  DisplayContents(setInt1);

  msetInt6.insert(-1);  // duplicate
  std::cout << "Contents of the multiset: " << std::endl;
  DisplayContents(msetInt6);

  std::cout << "Number of instances of '-1' in the multiset are: ";
  // std::multiset.count(value) 统计集合中 value 的个数
  std::cout << msetInt6.count(-1) << std::endl;
  
  return 0;
}

// $ g++ -o main 19.1_set.cpp 
// $ ./main.exe

// Contents of the set:      
// -1 24 42 2020 2021        
// Contents of the multiset: 
// -1 -1 24 42 2020 2021
// Number of instances of '-1' in the multiset are: 2