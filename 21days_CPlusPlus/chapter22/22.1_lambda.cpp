/**lambda 表达式
 * lambda 表达式一种定义匿名函数对象的简洁方法， C++11
 * C++14 新增泛型 lambda 表达式，支持 auto
 * 
 * 将 lambda 表达式视为包含公有 operator() 的匿名结构或者类，
 * 从这个角度看，lambda 表达式就是一个函数对象
 * 
 * lambda 表达式也称之为 lambda 函数
 * 
 * lambda 必须以 [] 开头，告诉编译器这是一个 lambda expression，
 * 接下来就是参数列表，提供给 operator() 的参数列表一致
 * 然后就是 lambda expression 的具体实现代码，函数体内容
 * 
 * [] (paraType parameter list) { lambda expression code;}
 * [] (paraType& parameter list) { lambda expression code;}
 */

#include <iostream>
#include <algorithm>
#include <list>
#include <vector>

int main(int argc, char** argv)
{
  std::vector<int> numInVec {101, -4, 500, 21, 42, -1};
  std::list<char> charInList {'s', 'h', 'z', 'd', 'l'};

  std::cout << "Display elements in a vector using a lambda expression: " << std::endl;
  std::cout << "=====================================================" << std::endl;

  // diaplay the array of intergers.
  // 一元函数对应的 lambda 表达式
  std::for_each(numInVec.cbegin(), numInVec.cend(),
                [] (const int& element) {std::cout << element << ' ';});
  std::cout << std::endl << "=====================================================" << std::endl;

  // diaplay the list of char.
  // C++14 支持使用 auto 在 lambda expression
  std::for_each(charInList.cbegin(), charInList.cend(),
                [] (auto& element) {std::cout << element << ' ';});
  std::cout << std::endl << "=====================================================" << std::endl;
  
  return 0;
}

// $ g++ -o mian 22.1_lambda.cpp 
// $ ./mian.exe 

// Display elements in a vector using a lambda expression: 
// =====================================================   
// 101 -4 500 21 42 -1
// =====================================================   
// s h z d l
// ===================================================== 
