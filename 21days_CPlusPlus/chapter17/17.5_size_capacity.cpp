#include <iostream>
#include <vector>

// std::vector 大小指的是实际储存的元素数量，size() 返回
// std::vector 容量指的是在重新分配内存之前能够储存的元素数量， capacity()
// size <= capacity
// 为了避免高频的重新分配内存，使用成员函数 reserve(number) 来一次增加内存
// 在重新分配 vector 内部缓冲区时提前增加容量方面，C++ 标准没有做任何规定

int main(int argc, char** argv)
{
  // instantiate a vector object that holds 5 intergers of default value.
  std::vector<int> integers (5);

  std::cout << "Vector of integers was instantiate with " << std::endl;
  std::cout << "Size: " << integers.size();
  std::cout << " , Capacity: " << integers.capacity() << std::endl;

  // inserting a 6th element in to the vector.
  integers.push_back(42);

  std::cout << "After inserting an additional element..." << std::endl;
  std::cout << "Size: " << integers.size() << " , Capacity: " << integers.capacity() << std::endl;

  // inserting another element.
  integers.push_back(24);
  
  std::cout << "After inserting another additional element..." << std::endl;
  std::cout << "Size: " << integers.size() << " , Capacity: " << integers.capacity() << std::endl;
  
  return 0;
}

// $ g++ -o main 17.5_size_capacity.cpp 
// $ ./main.exe 

// Vector of integers was instantiate with      
// Size: 5 , Capacity: 5
// After inserting an additional element...     
// Size: 6 , Capacity: 10
// After inserting another additional element...
// Size: 7 , Capacity: 10