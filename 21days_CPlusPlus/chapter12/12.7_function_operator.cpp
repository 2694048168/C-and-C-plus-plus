#include <iostream>
#include <string>

/**函数运算符 operator()
 * operator() 让对象类似函数一样，被称之为函数运算符
 * 常用于标准模板库 STL 中的算法部分，
 * 根据操作数的数量，这样的函数对象被称之为单目谓词或者双目谓词
 */

class Display
{
public:
  void operator () (std::string input) const
  {
    std::cout << input << std::endl;
  }
};


int main(int argc, char** argv)
{
  Display displayFunctionObject;

  // equivalent to displayFunctionObject.operator () ("Display this strirng! ");
  displayFunctionObject ("This allows object to be called like a function.");
  displayFunctionObject ("Display this strirng! ");
  // 这样使得类对象可用类似函数一样的调用
  
  return 0;
}

// $ g++ -o main 12.7_function_operator.cpp 
// $ ./main.exe
// This allows object to be called like a function.
// Display this strirng!