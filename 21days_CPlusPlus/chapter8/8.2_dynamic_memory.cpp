#include <iostream>
#include <string>

int main(int argc, char** argv)
{
  // request for memory to store for an int 
  int* ptrToAge = new int;

  // using the allacate memory to store a integer
  std::cout << "Enter your dog's age: ";
  std::cin >> *ptrToAge;

  // using indirection operator * to access value
  std::cout << "Age " << *ptrToAge << " is stored at 0x" << std::hex << ptrToAge << std::endl;

  // delete ptrToAge to release memory
  delete ptrToAge;

  std::cout << "=====================" << std::endl;

  std::cout << "How many integeres shall I reserve memory for?" << std::endl;
  int numEntries = 0;
  std::cin >> numEntries;

  // int* numbers = new int[numEntries]; 申请内存，可能有失败的情况
  int* numbers = new int[numEntries];
  if (numbers)
  {
    std::cout << "Memeory allcocated at: 0x" << numbers << std::hex << std::endl;

    delete[] numbers;
  }
  else
  {
    std::cout << "The memory new failure." << std::endl;
  }

  return 0;
}


// $ g++ -o main 8.2_dynamic_memory.cpp 
// $ ./main.exe 
// Enter your dog's age: 34
// Age 34 is stored at 0x0x176e50
// =====================
// How many integeres shall I reserve memory for?
// 45
// Memeory allcocated at: 0x0x171440