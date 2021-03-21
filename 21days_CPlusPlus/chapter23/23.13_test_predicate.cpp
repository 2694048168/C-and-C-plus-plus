#include <iostream>
#include <algorithm>
#include <string>

// binary predicate
struct CaseInsensitiveCompare
{
  bool operator () (const std::string& str1, const std::string& str2)
  {
    std::string str1Copy (str1);
    std::string str2Copy (str2);

    std::transform(str1Copy.begin(), str1Copy.end(), str1Copy.begin(), tolower);
    std::transform(str2Copy.begin(), str2Copy.end(), str2Copy.begin(), tolower);

    return (str1Copy < str2Copy);
  }
};

int main(int argc, char** argv)
{
  std::string str1 {"LiWei"};
  std::string str2 {"Jxufe Software"};

  std::cout << "The first string: " << str1 << std::endl;
  std::cout << "The second string: " << str2 << std::endl;

  std::cout << "-------------------------------" << std::endl;
  std::cout << std::boolalpha << (str1 < str2) << std::endl;
  std::cout << std::boolalpha << (str2 < str1) << std::endl;

  std::cout << "-------------------------------" << std::endl;
  auto element = CaseInsensitiveCompare();
  std::cout << std::boolalpha << element(str1, str2) << std::endl;
  std::cout << std::boolalpha << element(str2, str1) << std::endl;

  return 0;
}
// TEST failure
// $ g++ -o main 23.13_test_predicate.cpp 
// $ ./main.exe

// The first string: LiWei
// The second string: Jxufe Software
// -------------------------------
// false
// true
// -------------------------------
// false
// true
