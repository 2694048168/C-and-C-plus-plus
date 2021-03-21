// Preprocessor directive that includes header iostream
// 预处理器通过 include 命令将 iostream 文件全部拷贝到这里
#include <iostream>

// Function declaration and definition
// 声明并定义函数
int DemoConsoleOutput()
{
  std::cout << "This is a simple string literal" << std::endl;
  std::cout << "Writing number five: " << 5 << std::endl;
  std::cout << "Performing division 10 / 5 = " << 10 / 5 << std::endl;
  std::cout << "Pi when approximated is 22 / 7 = " << 22 / 7 << std::endl;
  std::cout << "Pi is 22 / 7 = " << 22.0 / 7 << std::endl;

  return 0;
}

// Start of your program: the main function block
// 整个程序只能有一个主程序入口 main 函数，程序启动后就会跳到这里开始
int main(int argc, char**argv)
{
  // Call Function with return used the exit
  return DemoConsoleOutput();
}
