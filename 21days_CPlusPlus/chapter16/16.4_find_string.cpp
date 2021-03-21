#include <iostream>
#include <string>

// 在 std::string 中查找字符或子字符串
// 成员函数 find()

int main(int argc, char** argv)
{
  std::string sampleStr ("Good day string. Today is beautiful");
  std::cout << "Sample string is: " << std::endl << sampleStr << std::endl;
  std::cout << "============================================" << std::endl;

  // find substring "day" - find() returns position
  // typedef unsigned long long size_t
  // param1: substring; param2: offset 偏移量;
  size_t charPosition = sampleStr.find("day", 0);

  // check if the substring was found
  // found return the position;
  // not found return td::string::npos
  if (charPosition != std::string::npos)
  {
    std::cout << "First instance \"day\" at position: " << charPosition << std::endl;
  }
  else
  {
    std::cout << "The substring not found." << std::endl;
  }

  std::cout << "===========================================" << std::endl;
  std::cout << "Locating all instance of substring \"day\"" << std::endl;
  size_t substringPos = sampleStr.find("day", 0);
  while (substringPos != std::string::npos)
  {
    std::cout << "\"day\" found at position " << substringPos << std::endl;

    // make find() search forward from the next character onwards.
    size_t searchOffset = substringPos + 1;

    substringPos = sampleStr.find("day", searchOffset);
  }
  
  return 0;
}

// $ g++ -std=c++11 -o main 16.4_find_string.cpp 
// $ ./main.exe 

// Sample string is:
// Good day string. Today is beautiful
// ============================================
// First instance "day" at position: 5
// =========================================== 
// Locating all instance of substring "day"    
// "day" found at position 5
// "day" found at position 19