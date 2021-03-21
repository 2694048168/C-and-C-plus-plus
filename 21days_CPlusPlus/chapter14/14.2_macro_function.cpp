#include <iostream>
#include <string>

/**预处理器和编译器
 * 预处理器只是做简单的文本替换，不做安全类型检查
 * 故此做宏函数等表达式，使用括号进行约束计算优先级，
 * 否则文本替换会出现超出预期的结果，破坏了编程逻辑
 * 
 * 使用 assert 宏验证表达式
 * 断言 测试，调试模式下，发布模式一般禁用
 * 
 * 使用预处理指令和宏避免头文件的彼此包含而导致的递归问题
 * #ifndef、#ifdef、#endif
 * 这也是 C++ 中使用最多的宏功能之一
 */
/*
#ifndef file_name
#define file_name

#include <file_name.h>
// the file ......

#endif // file_name
*/

// 宏函数
// 宏不支持任何形式的类型安全
// 建议使用内联函数达到减少开销的宏函数一样功效
#define SQUARE(x) ((x) * (x))
#define PI 3.14159265
#define AREA_CIRCLE(r) (PI * (r) * (r))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

int main(int argc, char** argv)
{
  std::cout << "Please enter an integer: ";
  int num = 0;
  std::cin >> num;

  std::cout << "SQUARE(" << num << ") = " << SQUARE(num) << std::endl;
  std::cout << "Area of a cirlce with radius " << num << " is: " << AREA_CIRCLE(num) << std::endl;

  std::cout << "Please enter another integer: ";
  int num2 = 0;
  std::cin >> num2;

  std::cout << "MIN(" << num << ", " << num2 << ") = " << MIN(num, num2) << std::endl;
  std::cout << "MAX(" << num << ", " << num2 << ") = " << MAX(num, num2) << std::endl;
  
  return 0;
}

// $ g++ -o main 14.2_macro_function.cpp 
// $ ./main.exe 
// Please enter an integer: 6
// SQUARE(6) = 36
// Area of a cirlce with radius 6 is: 113.097
// Please enter another integer: 45
// MIN(6, 45) = 6 
// MAX(6, 45) = 45