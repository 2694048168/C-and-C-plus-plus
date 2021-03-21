#include <iostream>
#include <string>

/**基于模板的 STL string 实现
 * 模板类 std::basic_string<T>
 * 具体化 std::string; std::wstring (支持 Unicode 字符集)
 * 
 * C++14 中新增 operator ""s
 * C++17 中新增 std::string_view
 * C++ standard website: https://isocpp.org/std/status
 */


int main(int argc, char** argv)
{
  using namespace std;

  string str1("Traditional string \0 initialization");
  cout << "Str1: " << str1 << " Length: " << str1.length() << endl;

  string str2("C++14 \0 initialization using literals"s);
  cout << "Str2: " << str2 << " Length: " << str2.length() << endl;

 return 0;
}

// $ g++ -std=c++14 -o main 16.8_operators_view.cpp 
// $ ./main.exe 
// Str1: Traditional string  Length: 19
// Str2: C++14   initialization using literals Length: 37