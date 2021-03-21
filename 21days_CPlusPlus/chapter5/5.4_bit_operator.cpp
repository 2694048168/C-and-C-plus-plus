#include <iostream>
#include <bitset>

int main(int argc, char** argv)
{
  std::cout << "Enter a number ( 0 - 255 ): ";
  unsigned short inputNum = 0;
  std::cin >> inputNum;

  std::bitset<8> inputBits(inputNum);
  std::cout << inputNum << " in binary is " << inputBits << std::endl;

  std::bitset<8> bitwiseNOT = (~inputNum);
  std::cout << "Logical NOT ~ " << std::endl;
  std::cout << "~" << inputBits << " = " << bitwiseNOT << std::endl;

  std::cout << "Logical AND & with 00001111" << std::endl;
  std::bitset<8> bitwiseAND = (0X0F & inputNum);
  std::cout << "00001111 & " << inputBits << " = " << bitwiseAND << std::endl;

  std::cout << "Logical OR | with 00001111" << std::endl;
  std::bitset<8> bitwiseOR = (0X0F | inputNum);
  std::cout << "00001111 | " << inputBits << " = " << bitwiseOR << std::endl;

  std::cout << "Logical XOR & with 00001111" << std::endl;
  std::bitset<8> bitwiseXOR = (0X0F & inputNum);
  std::cout << "00001111 ^ " << inputBits << " = " << bitwiseXOR << std::endl;

  // 位运算符 与-& 或-| 取反-~ 异或-^
  // 常用于二进制等一些位操作应用中，加密等算法中
  // 嵌入式中常常用位操作进行对某些位进行屏蔽等常规操作

  // 按位右移运算符 >> ，右移一位效果等同于除以 2，右移 n 位效果是除以 2^n
  // 按位左移运算符 << ，左移一位效果等同于乘以 2，左移 n 位效果是乘以 2^n

  std::cout << "================================" << std::endl;
  std::cout << "Enter a number: ";
  int userNum = 0;
  std::cin >> userNum;

  std::cout << "Quarter: " << (userNum >> 2) << std::endl;
  std::cout << "Half: " << (userNum >> 1) << std::endl;
  std::cout << "Double: " << (userNum << 1) << std::endl;
  std::cout << "Quadruple: " << (userNum << 2) << std::endl;

  return 0;
}


// $ g++ -o main 5.4_bit_operator.cpp 
// $ ./main.exe 
// Enter a number ( 0 - 255 ): 25
// 25 in binary is 00011001      
// Logical NOT ~
// ~00011001 = 11100110
// Logical AND & with 00001111   
// 00001111 & 00011001 = 00001001
// Logical OR | with 00001111    
// 00001111 | 00011001 = 00011111
// Logical XOR & with 00001111
// 00001111 ^ 00011001 = 00001001
// ================================
// Enter a number: 2
// Quarter: 0
// Half: 1
// Double: 4
// Quadruple: 8

// $ ./main.exe 
// Enter a number ( 0 - 255 ): 25
// 25 in binary is 00011001
// Logical NOT ~
// ~00011001 = 11100110
// Logical AND & with 00001111
// 00001111 & 00011001 = 00001001
// Logical OR | with 00001111
// 00001111 | 00011001 = 00011111
// Logical XOR & with 00001111
// 00001111 ^ 00011001 = 00001001
// ================================
// Enter a number: 8
// Quarter: 2
// Half: 4
// Double: 16
// Quadruple: 32