#include <iostream>

int main(int argc, char**argv)
{
  std::cout << "Hello World!" << std::endl;

  return 0;
}


/**Note
 * 编译命令：g++ -o hello 1.1_Hello.cpp
 * 
 * 理解编译错误
 * hello.cpp(6): error C2143: syntax error : missing ';' before 'return'
 * 出错文件(行数) 错误编号      错误类型(语法)        编译器分析的错误
 */