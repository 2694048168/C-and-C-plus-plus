#include <iostream>
#include <iomanip>
#include <string>

// 读取应该注意并考虑到缓冲区的情况！！！
int main(int argc, char** argv)
{
  // 1. 使用 std::cin 将输入读取到基本类型变量中
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "Please enter an integer: ";
  int inputNum = 0;
  std::cin >> inputNum;
  std::cout << inputNum << std::endl;

  std::cout << "Please enter three characters separated by space: " << std::endl;
  char char1 = '\0', char2 = '\0', char3 = '\0';
  std::cin >> char1 >> char2 >> char3;
  std::cout << char1 << char2 << char3 << std::endl;

  // 2. 使用 std::cin:get 将输入读取到 char* 缓冲区中
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "Please enter a line: " << std::endl;
  char charBuffer[10] = {0};
  std::cin.get(charBuffer, 9);
  std::cout << "charBuffer: " << charBuffer << std::endl;

  // 3. 使用 std::cin 将输入读取到 std::string 中
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "Please enter your name: " << std::endl;
  std::string name;
  std::cin >> name;
  std::cout << "Hello, " << name << std::endl;

  // 4. 使用 getline 读取整行用户输入
  std::cout << "--------------------------------------------" << std::endl;
  std::cout << "Please enter your name: " << std::endl;
  std::string yourName;
  getline(std::cin, yourName);
  std::cout << "Hi, " << yourName << std::endl;

  return 0;
}

// $ g++ -o main 27.4_cin.cpp 
// $ ./main.exe 
// --------------------------------------------
// Please enter an integer: 42
// 42
// Please enter three characters separated by space: 
// liweijxufe
// liw
// --------------------------------------------
// Please enter a line:
// charBuffer: eijxufe
// --------------------------------------------
// Please enter your name:
// li wei
// Hello, li
// --------------------------------------------
// Please enter your name:
// Hi,  wei
