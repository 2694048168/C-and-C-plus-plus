/**使用流进行输入和输出
 * C++ 流就是读写(输入和输出)逻辑的通用实现，同一模式读写数据
 * 屏蔽读写的物理设备和操作系统，抽象出流的概念同一读写逻辑
 * C++ 提供一组标准类和头文件：std命令空间中的 C++ 流类；std命名空间中的流控制符 
 */

#include <iostream> // 流类
#include <iomanip> // 流控制符

int main(int argc, char** argv)
{
  std::cout << "Please enter an integer: ";
  int input;
  std::cin >> input;

  std::cout << "Integer in octal: " << std::oct << input << std::endl;
  std::cout << "Integer in hexadecimal: " << std::hex << input << std::endl;
  std::cout << "Integer in hex using base notation: " << std::setiosflags(
    std::ios_base::hex|std::ios_base::showbase|std::ios_base::uppercase) << input << std::endl;
  std::cout << "Integer after resetting I/O flags: " << std::resetiosflags(
    std::ios_base::hex|std::ios_base::showbase|std::ios_base::uppercase) << input << std::endl;
  
  return 0;
}

// $ g++ -o main 27.1_stream.cpp 
// $ ./main.exe 

// Please enter an integer: 255
// Integer in octal: 377
// Integer in hexadecimal: ff
// Integer in hex using base notation: 0XFF
// Integer after resetting I/O flags: 255 