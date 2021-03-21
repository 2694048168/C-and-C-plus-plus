#include <iostream>
#include <iomanip>

// 使用 setw 对其文本和设置字段宽度
int main(int argc, char** argv)
{
  std::cout << "Hey - default." << std::endl;
  std::cout << std::setw(35) << "Hey - right aligned." << std::endl;
  std::cout << std::setw(35) << std::setfill('*') << "Hey - right aligned." << std::endl;
  std::cout << "Hey - back to default." << std::endl;
  
  return 0;
}

// $ g++ -o mian 27.3_setw.cpp 
// $ ./mian.exe 

// Hey - default.
//                Hey - right aligned.
// ***************Hey - right aligned.
// Hey - back to default.