#include <iostream>
#include <string>
#include <tuple>

template <typename tupleType>
void DisplayTupleInfo(tupleType& tup)
{
  const int numMenbers = std::tuple_size<tupleType>::value;
  std::cout << "Num elements in tuple: " << numMenbers << std::endl;
  // 使用 std::get 一种访问元组的每一个元素的机制
  std::cout << "Last element value: " << std::get<numMenbers -1> (tup) << std::endl;
}

// 元组是一个高级概念，常用于通用模板编程
int main(int argc, char** argv)
{
  std::tuple<int, char, std::string> tup1(std::make_tuple(101, 's', "Hello Tuple."));
  DisplayTupleInfo(tup1);

  auto tup2(std::make_tuple(3.14, false));
  DisplayTupleInfo(tup2);

  auto concatTup(std::tuple_cat(tup2, tup1));
  DisplayTupleInfo(concatTup);

  double PI;
  std::string sentence;
  // std::tie 将元组的内容拆封(复制)到对象中
  // std::ignore 让 std::tie 忽略不感兴趣的元组成员
  std::tie(PI, std::ignore, std::ignore, std::ignore, sentence) = concatTup;
  std::cout << "Unpacked! PI: " << PI << " and \"" << sentence << "\"" << std::endl;
  
  return 0;
}

// $ g++ -std=c++11 -o main 14.8_template_tuple.cpp       
// $ ./main.exe 
// Num elements in tuple: 3
// Last element value: Hello Tuple.
// Num elements in tuple: 2
// Last element value: 0
// Num elements in tuple: 5
// Last element value: Hello Tuple.
// Unpacked! PI: 3.14 and "Hello Tuple."