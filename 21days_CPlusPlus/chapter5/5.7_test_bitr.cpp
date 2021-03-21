#include <iostream>
#include <bitset>

int main(int argc, char** argv)
{
  std::cout << "Enter bool value ( 0 or 1 ): ";
  bool inputNum = false;
  std::cin >> inputNum;

  std::cout << "Enter another bool value ( 0 or 1 ): ";
  bool inpuValue = false;
  std::cin >> inpuValue;

  std::cout << "==============" << std::endl;

  bool bitwiseNOT = (~inputNum);
  std::cout << "Logical NOT ~ " << std::endl;
  std::cout << "~" << inputNum << " = " << bitwiseNOT << std::endl;
  std::cout << "==============" << std::endl;

  std::cout << "Logical AND & with inpuValue" << std::endl;
  bool bitwiseAND = (inpuValue & inputNum);
  std::cout << "inpuValue & inputNum = " << bitwiseAND << std::endl;
  std::cout << "==============" << std::endl;

  std::cout << "Logical OR | with inpuValue" << std::endl;
  bool bitwiseOR = (inpuValue | inputNum);
  std::cout << "inpuValue | inputNum = " << bitwiseOR << std::endl;
  std::cout << "==============" << std::endl;

  std::cout << "Logical XOR & with inpuValue" << std::endl;
  bool bitwiseXOR = (inpuValue & inputNum);
  std::cout << "inpuValue ^ inputNum = " << bitwiseXOR << std::endl;

  return 0;
}

// $ g++ -o main 5.7_test_bitr.cpp 
// $ ./main.exe
// Enter bool value ( 0 or 1 ): 1
// Enter another bool value ( 0 or 1 ): 0
// ==============
// Logical NOT ~
// ~1 = 1
// ==============
// Logical AND & with inpuValue
// inpuValue & inputNum = 0
// ==============
// Logical OR | with inpuValue
// inpuValue | inputNum = 1
// ==============
// Logical XOR & with inpuValue
// inpuValue ^ inputNum = 0