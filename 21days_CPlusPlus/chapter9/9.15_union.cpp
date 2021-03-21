#include <iostream>

/**共用体 union
 * 一种特殊的数据存储机制
 * 一种特殊的类，每次只有一个非静态数据成员处于活动状态
 * 共用体的数据成员默认为共有的，不能继承
 */
union SimpleUnion
{
  int num;
  char alphabet;
};

struct ComplexType
{
  enum DataType
  {
    Int,
    Char
  }Type;

  union Value
  {
    int num;
    char alphabet;

    Value() {}
    ~Value() {}
  }value;
};

void DisplayComplexType(const ComplexType& obj)
{
  switch (obj.Type)
  {
  case ComplexType::Int:
    std::cout << "Union contains numbers: " << obj.value.num << std::endl;
    break;

  case ComplexType::Char:
    std::cout << "Union contains character: " << obj.value.alphabet << std::endl;
    break;
  
  default:
    break;
  }

  return;
}

int main(int argc, char**)
{
  SimpleUnion unionOne, unionTwo;
  unionOne.num = 2021;
  unionTwo.alphabet = 'C';
  std::cout << "sizeof(unionOne) containing integer: " << sizeof(unionOne) << std::endl;
  std::cout << "sizeof(unionTwo) containing character: " << sizeof(unionTwo) << std::endl;

  ComplexType myDataOne, myDataTwo;
  myDataOne.Type = ComplexType::Int;
  myDataOne.value.num = 2020;
  myDataOne.Type = ComplexType::Char;
  myDataOne.value.alphabet = 'X';

  DisplayComplexType(myDataOne);
  DisplayComplexType(myDataTwo);

  return 0;
}


// $ g++ -o main 9.15_union.cpp 
// $ ./main.exe 
// sizeof(unionOne) containing integer: 4  
// sizeof(unionTwo) containing character: 4
// Union contains character: X