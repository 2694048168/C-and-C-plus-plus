#include <iostream>

int main(int argc, char**argv)
{
  auto coinFlippedHeads = true;
  // auto 自动推断变量类型，必须初始化变量
  auto largeNumber = 2500000000000000;

  std::cout << "coinFippedHeads = " << coinFlippedHeads << " , sizeof = "
            << sizeof(coinFlippedHeads) << std::endl;

  std::cout << "largeNumber = " << largeNumber << " , sizeof = "
            << sizeof(largeNumber) << std::endl;

  return 0;
}

// $ g++ -o main 3.6_auto_infer.cpp 
// $ ./main
// coinFippedHeads = 1 , sizeof = 1
// largeNumber = 2500000000000000 , sizeof = 8

// 使用 typedef 替换变量类型
// typedef old_long_variable_type new_short_name;
// typedef unsigned int strictly_positive_integer;
// strictly_positive_integer numEggsInBasket = 4532;