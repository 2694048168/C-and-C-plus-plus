#include <iostream>
#include <string>
#include <algorithm>

// 截短 STL string
// string class given clear() to 清除全部内容并重置 string 对象
// STL String 提供 erase() function
// 1. 在给定指定偏移位置和字符数时候删除指定数目的字符
// string sampleStr ("Hello String! Wake up to a beautiful day!");
// sampleStr.erase (13, 28); // Hello String!

// 2. 在给定指向字符的迭代器时删除该字符
// sampleStr.erase (iCharS); // iterator points to a specific character

// 3. 在给定由两个迭代器指定的范围时删除该范围的字符
// sampleStr.erase (sampleStr.begin (), sampleStr.end ()); // erase from begin to end

int main(int argc, char** argv)
{
  std::string sampleStr ("Hello String. Wake up to a beautiful day.");
  std::cout << "The original sample string is: " << std::endl;
  std::cout << sampleStr << std::endl << std::endl;

  // Delete characters given position and count.
  std::cout << "Truncating the second sentence: "<< std::endl;
  sampleStr.erase(13, 28);
  std::cout << sampleStr << std::endl << std::endl;

  // find character 'S' using find() algorithm
  std::string::iterator iCharS = find(sampleStr.begin(), sampleStr.end(), 'S');

  // if character found, 'erase' to delete a character
  std::cout << "Erasing character 'S' from the sample string:" << std::endl;
  if (iCharS != sampleStr.end())
  {
    sampleStr.erase(iCharS);
  }
  std::cout << sampleStr << std::endl << std::endl;

  // erase a range of characters using an overloaded version of erase()
  std::cout << "Erasing a range between begin() and end():" << std::endl;
  sampleStr.erase(sampleStr.begin(), sampleStr.end());

  // verify the length after the erase() operator above.
  // if (sampleStr.length() == 0)
  if (!sampleStr.length())
  {
    std::cout << "The string is empty." << std::endl;
  }
  
  return 0;
}

// $ g++ -std=c++11 -o main 16.5_truncate_string.cpp 
// $ ./main.exe 

// The original sample string is: 
// Hello String. Wake up to a beautiful day.

// Truncating the second sentence:
// Hello String.

// Erasing character 'S' from the sample string:
// Hello tring.

// Erasing a range between begin() and end():
// The string is empty.