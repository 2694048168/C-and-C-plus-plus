#include <iostream>

int main(int argc, char** argv)
{
  std::cout << "Enter true(1) or false(0) for two operands: ";
  bool operatorOne = false, operatorTwo = false;
  std::cin >> operatorOne;
  std::cin >> operatorTwo;

  std::cout << operatorOne << " AND " << operatorTwo << " = " 
            << (operatorOne && operatorTwo) << std::endl;
  std::cout << operatorOne << " OR " << operatorTwo << " = " 
            << (operatorOne || operatorTwo) << std::endl;
  std::cout << operatorOne << " NOT " << " = " << (!operatorOne) << std::endl;
  std::cout << operatorTwo << " NOT " << " = " << (!operatorTwo) << std::endl;

  // 逻辑运算符 与-&& 或-|| 非-！
  // 常用于 if 等条件判断分支语句中
  // 使用逻辑运算帮助判断是否购买汽车
  std::cout << "Answer questions with 0 or 1." << std::endl;
  std::cout << "Is there a discount on your favorite car? ";
  bool onDiscount = false;
  std::cin >> onDiscount;

  std::cout << "Did you get a fantastic bonus? ";
  bool fantasticBonus = false;
  std::cin >> fantasticBonus;

  if (onDiscount || fantasticBonus)
  {
    std::cout << "Congratulations, you can buy that car!" << std::endl;
  }
  else
  {
    std::cout << "Sorry, waiting a while is a good idea." << std::endl;
  }

  if (!onDiscount)
  {
    std::cout << "Car not on discount." << std::endl;
  }

  return 0;
}


// $ g++ -o main 5.3_logic_operation.cpp 
// $ ./main.exe 
// Enter true(1) or false(0) for two operands: 0
// 1
// 0 AND 1 = 0
// 0 OR 1 = 1
// 0 NOT  = 1
// 1 NOT  = 0
// Answer questions with 0 or 1.
// Is there a discount on your favorite car? 0
// Did you get a fantastic bonus? 1
// Congratulations, you can buy that car!
// Car not on discount.