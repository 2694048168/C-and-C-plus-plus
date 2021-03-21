/** STL string class
 * STL 提供一个用于字符串操作的容器类，string
 * string 能够动态调整大小，提供辅助函数和操作字符串方法
 * STL 中的都是标准的，经过测试的可移植的功能
 * 减少一些操作的函数以及内存管理分配细节
 * 
 * STL string class
 * 1. 复制
 * 2. 连接
 * 3. 查找字符和子字符串
 * 4. 截短
 * 5. 使用标准模板库提供的算法实现字符串的反转和大小写转换
 */

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  const char* constCStyleString = "Hello String.";
  std::cout << "Constant string is: " << constCStyleString << std::endl;

  // constructor
  std::string strFromConst (constCStyleString);
  std::cout << "strFromConst is: " << strFromConst << std::endl;

  std::string str2 ("Hello String.");
  std::string str2Copy(str2);
  std::cout << "str2Copy is: " << str2Copy << std::endl;

  // initialize a string to the first 5 characters of another
  std::string strPartialCopy(constCStyleString, 5);
  std::cout << "strPartialCopy is: " << strPartialCopy << std::endl;

  // initialize a string object to contain 10 'a's
  std::string strRepeatChars(10, 'a');
  std::cout << "strRepeatChars is: " << strRepeatChars << std::endl;
  
  return 0;
}

// $ touch 16.1_string.cpp
// $ g++ -std=c++11 -o main 16.1_string.cpp 
// $ ./main.exe 

// Constant string is: Hello String.
// strFromConst is: Hello String.   
// str2Copy is: Hello String.       
// strPartialCopy is: Hello
// strRepeatChars is: aaaaaaaaaa  