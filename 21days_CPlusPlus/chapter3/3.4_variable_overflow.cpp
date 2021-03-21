#include <iostream>

int main(int argc, char**argv)
{
  unsigned short shortValue = 65535;
  std::cout << shortValue << std::endl;
  std::cout << "Incrementing unsigned short " << shortValue << " gives: "
            << ++shortValue << std::endl;

  short signedShort = 32767;
  std::cout << signedShort << std::endl;
  std::cout << "Incrementing signed short " << signedShort << " gives: "
            << ++signedShort << std::endl;

  // 溢出导致程序的无法预测
  return 0;
}