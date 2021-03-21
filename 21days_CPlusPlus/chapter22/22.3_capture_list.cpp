#include <iostream>
#include <algorithm>
#include <vector>

// 通过 捕获列表 capture list 接受 状态变量 的 lambda 表达式
// 状态变量的方式传递参数，该参数列表称之为捕获列表 capture list
// 状态变量参数列表等效于函数对象类或者结构中的数据成员

/**lambda expression 通用语法
 * 1. 方括号开头，即捕获列表里面是状态变量列表参数；
 * 2. 如果需要在 lambda expression 中修改状态变量的值，需要使用 mutable
 * 3. 如果需要修改后的状态变量的值在外部生效，需要使用引用传递状态变量参数
 * 4. 圆括号继续，即函数对象的参数列表
 * 5. 大括号接着，即函数体内容，如果有多条语句，必须显示指定返回类型
 * 6. 通过尾置返回类型方法，显式指定 lambda 函数返回类型
 * 
 * [&stateVar1, &stateVar2, ] (Type& parameter list) mutable {; ; ;} -> returnType
 * 
 */

int main(int argc, char** argv)
{
  std::vector<int> numInVec {21, 25, 42, 52, 2, 99};
  std::cout << "The vector contains: {24, 25, 42, 52, 2, 99}" << std::endl;
  std::cout << "Please enter divisor ( > 0 ): ";
  int divisor = 2;
  std::cin >> divisor;

  // find the first element that is a multiple of divisor.
  std::vector<int>::iterator element;
  element = std::find_if(numInVec.begin(), numInVec.end(),
                         [divisor] (int dividend) {return (dividend % divisor) == 0;});

  if (element != numInVec.end())
  {
    std::cout << "First element in vector divisible by " << divisor << " : " << *element << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 22.3_capture_list.cpp 
// $ ./main.exe 

// The vector contains: {24, 25, 42, 52, 2, 99}
// Please enter divisor ( > 0 ): 4
// First element in vector divisible by 4 : 52