// Preprocessor directive that includes header iostream
// 预处理器通过 include 命令将 iostream 文件全部拷贝到这里
#include <iostream>

// Start of your program: the main function block
// 整个程序只能有一个主程序入口 main 函数，程序启动后就会跳到这里开始
int main(int argc, char**argv)
{
  // Write to the screen
  // 向标准输出写入
  std::cout << "Hello World!" << std::endl;

  // Return a value to the OS
  // 向操作系统返回一个值，便于操作系统做出响应
  return 0;
}


/**Note
 * 编译命令：g++ -o hello 2.1_HelloWorldAnalysis.cpp
 */