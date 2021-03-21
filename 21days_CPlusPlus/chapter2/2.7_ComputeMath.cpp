// Preprocessor directive that includes header iostream
// 预处理器通过 include 命令将 iostream 文件全部拷贝到这里
#include <iostream>

// Declare a function for C++
int DemoConsoleOutput();

// Start of your program: the main function block
// 整个程序只能有一个主程序入口 main 函数，程序启动后就会跳到这里开始
int main(int argc, char**argv)
{
  // Call invoke the function
  DemoConsoleOutput();

  // Return a value to the OS
  // 向操作系统返回一个值，便于操作系统做出响应
  return 0;
}

// Define implement thr previously declared function
int DemoConsoleOutput()
{
  std::cout << "This is a simple string literal" << std::endl;
  std::cout << "Writing number five: " << 5 << std::endl;
  std::cout << "Performing division 10 / 5 = " << 10 / 5 << std::endl;
  std::cout << "Performing division 10 * 5 = " << 10 * 5 << std::endl;
  std::cout << "Pi when approximated is 22 - 7 = " << 22 - 7 << std::endl;
  std::cout << "Pi when approximated is 22 + 7 = " << 22 + 7 << std::endl;
  std::cout << "Pi when approximated is 22 % 7 = " << 22 % 7 << std::endl;
  std::cout << "Pi when approximated is 22 / 7 = " << 22 / 7 << std::endl;
  std::cout << "Pi is 22 / 7 = " << 22.0 / 7 << std::endl;

  return 0;
}

// GCC 编译指令
// $ g++ -o main 2.7_ComputeMath.cpp 

// 执行结果
// $ ./main.exe                  
// This is a simple string literal    
// Writing number five: 5                                        
// Performing division 10 / 5 = 2     
// Performing division 10 * 5 = 50                            
// Pi when approximated is 22 - 7 = 15
// Pi when approximated is 22 + 7 = 29
// Pi when approximated is 22 % 7 = 1 
// Pi when approximated is 22 / 7 = 3 
// Pi is 22 / 7 = 3.14286