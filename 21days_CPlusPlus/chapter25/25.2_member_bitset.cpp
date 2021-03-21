#include <iostream>
#include <string>
#include <bitset>

// 使用 std::bitset 以及其成员函数
// std::bitset 的运算符
// std::bitset 缺点就是不能动态的调整长度，必须在编译阶段决定
// 为了克服，提供了 vector<bool> 或者 bit_vector
int main(int argc, char** argv)
{
  std::bitset<8> inputBits;
  std::cout << "Please enter a 8-bit sequence: ";
  std::cin >> inputBits;

  std::cout << "Num 1s you supplied: " << inputBits.count() << std::endl;
  std::cout << "Num 0s you supplied: " << inputBits.size() - inputBits.count() << std::endl;

  // copy
  std::bitset<8> inputFlipped(inputBits);
  // toggle the bits.
  inputFlipped.flip();
  std::cout << "Flipped version is: " << inputFlipped << std::endl;

  std::cout << "Result of AND, OR and XOR between the two: " << std::endl;
  std::cout << inputBits << " & " << inputFlipped << " = " << (inputBits & inputFlipped) << std::endl;
  std::cout << inputBits << " | " << inputFlipped << " = " << (inputBits | inputFlipped) << std::endl;
  std::cout << inputBits << " ^ " << inputFlipped << " = " << (inputBits ^ inputFlipped) << std::endl;

  return 0;
}

// $ g++ -o main 25.2_member_bitset.cpp 
// $ ./main.exe 

// Please enter a 8-bit sequence: 10100011
// Num 1s you supplied: 4
// Num 0s you supplied: 4
// Flipped version is: 01011100
// Result of AND, OR and XOR between the two: 
// 10100011 & 01011100 = 00000000
// 10100011 | 01011100 = 11111111
// 10100011 ^ 01011100 = 11111111