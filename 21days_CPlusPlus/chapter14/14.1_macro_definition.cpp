#include <iostream>
#include <string>

/**预处理器和编译器
 * 预处理器是在编译器之前，预处理编译指令都是以 # 开头
 * 预处理器只是做简单的文本替换，不做安全类型检查
 */

// 宏常量
// 建议使用 const 和数据类型进行定义常量，
// 避免宏简单的文本替换，而不检查类型
#define ARRAY_SIZE 12
#define PI 3.14159265
#define MY_DOUBLE double
#define FAV_WHISKY "Wei Li"

int main(int argc, char** argv)
{
  int numbers[ARRAY_SIZE] = {0};
  std::cout << "Array's length: " << sizeof(numbers) / sizeof(int) << std::endl;

  std::cout << "Please enter a radius: ";
  MY_DOUBLE radius = 0;
  std::cin >> radius;
  std::cout << "Area is: " << PI * radius * radius << std::endl;

  std::string favoriteWhisky(FAV_WHISKY);
  std::cout << "My favorite drink is: " << FAV_WHISKY << std::endl;
  
  return 0;
}


// $ g++ -o main 14.1_macro_definition.cpp 
// $ ./main.exe q
// Array's length: 12     
// Please enter a radius: 6
// Area is: 113.097
// My favorite drink is: Wei Li