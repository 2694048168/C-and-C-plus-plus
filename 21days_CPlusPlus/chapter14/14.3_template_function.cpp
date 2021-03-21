#include <iostream>
#include <string>

/**模板简介
 * 模板能够定义一种适用不同类型对象的行为，类似宏，
 * 但是宏不是类型安全的，而模板是类型安全的
 * 模板语法声明：
 * template <parameter list>
 * template function / class declaration
 * 
 * 模板函数
 * template <typename T1, typename T2 = T1>
 * bool TemplateFunction(const T1& param1, const T2& param2);
 * 
 * 模板类
 * template <typename T1, typename T2 = T1>
 * class MyTemplate
 * {
 *  public:
 *    T1 GetObj() {return member;}
 *    // the other members
 *  private:
 *    T1 number1;
 *    T2 number2;
 * };
 * 
 * 各种类型的模板声明
 * 1. 函数的声明和定义
 * 2. 类的定义和声明
 * 3. 类模板的成员函数或者成员类的声明或者定义
 * 4. 类模板的静态数据成员的定义
 * 5. 嵌套在类模板中的类的静态数据成员的定义
 * 6. 类或者模板的成员模板的定义
 */

// template function
template <typename Type>
const Type& GetMax(const Type& value1, const Type& value2)
{
  if (value1 > value2)
  {
    return value1;
  }
  else
  {
    return value2;
  }
}

// template function
template <typename Type>
void DisplayComaprison(const Type& value1, const Type& value2)
{
  std::cout << "GetMax(" << value1 << ", " << value2 << ") = " << GetMax(value1, value2) << std::endl;
}


int main(int argc, char** argv)
{
  int num1 = -101, num2 = 2021;
  DisplayComaprison(num2, num1);

  double d1 = 3.14, d2 = 3.1415;
  DisplayComaprison(d1, d2);

  std::string name1("Wei"), name2("Li");
  DisplayComaprison(name1, name2);
  
  return 0;
}

// $ g++ -o main 14.3_template_function.cpp 
// $ ./main.exe 
// GetMax(2021, -101) = 2021    
// GetMax(3.14, 3.1415) = 3.1415
// GetMax(Wei, Li) = Wei 