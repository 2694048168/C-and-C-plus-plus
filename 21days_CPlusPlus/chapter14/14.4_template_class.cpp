#include <iostream>
#include <string>

/**模板简介
 * 类是一种编程单元，函数也是一种编程单元
 * 类是一种设计蓝图，那么模板类就是蓝图的蓝图
 * 
 * 声明包含多个参数的模板类
 * template <typename Type1, typename T2>
 * class {};
 * 
 * 声明包含默认参数的模板类
 * template <typename Type=int, typename Type2=int>
 * class {};
 * 
 * 可以通过重用该模式针对不同的变量类型实现相同的逻辑，
 * 提高代码的复用性
 */

// template class
template <typename Type1, typename Type2>
class HoldsPair
{
public:
  HoldsPair(const Type1& val1, const Type2& val2) : value1(val1), value2(val2)
  {

  }

  const Type1& GetFirstValue() const
  {
    return value1;
  }

  const Type2& GetSecondValue() const
  {
    return value2;
  }

private:
  Type1 value1;
  Type2 value2;
};


int main(int argc, char** argv)
{
  // 可以通过重用该模式针对不同的变量类型实现相同的逻辑，
  HoldsPair<int, double> pairIntDbl (300, 10.09);
  HoldsPair<short,const char*> pairShortStr(25, "Learn templates, love C++");

  std::cout << "The first object contains -" << std::endl;
  std::cout << "Value 1: " << pairIntDbl.GetFirstValue () << std::endl;
  std::cout << "Value 2: " << pairIntDbl.GetSecondValue () << std::endl;

  std::cout << "=======================" << std::endl;
  std::cout << "The second object contains -" << std::endl;
  std::cout << "Value 1: " << pairShortStr.GetFirstValue () << std::endl;
  std::cout << "Value 2: " << pairShortStr.GetSecondValue () << std::endl;
  
  return 0;
}

// $ g++ -o main 14.4_template_class.cpp 
// $ ./main.exe
// The first object contains -
// Value 1: 300
// Value 2: 10.09
// =======================
// The second object contains -
// Value 1: 25
// Value 2: Learn templates, love C++