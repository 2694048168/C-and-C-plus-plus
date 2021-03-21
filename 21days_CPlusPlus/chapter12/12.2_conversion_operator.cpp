#include <iostream>
#include <string>
#include <sstream>

// 转换运算符
// 为了类对象可以用于 std::cout 等流对象，需要添加一个 返回 const char* 的运算符
// operator const char* ()
// {
//   // operator implementation thar return a char*
// }

class Date
{
public:
  // constructor using list initialization
  Date(int inputMonth, int inputDay, int inputYear)
      : month(inputMonth), day(inputDay), year(inputYear)
  {

  }

  // operator const char*() 转换运算符，便于类对象的使用
  // 这样后在 std::cout 中能够直接使用 类对象 了
  // 使用 explicit 要求使用强制类型转换来确认转换意图
  explicit operator const char*()
  // operator const char*()
  {
    // assists string construction
    std::ostringstream formatetedDate;
    formatetedDate << month << " / " << day << " / " << year;

    dateInstring = formatetedDate.str();
    return dateInstring.c_str();
  }

private:
  int day, month, year;
  std::string dateInstring;
};

int main(int argc, char** argv)
{
  Date hoilday (3, 24, 2021);
  // 为什么类对象可以直接在标准输出流中使用，因为有了 转换运算符
  // std::cout << "hoilday is on: " << hoilday << std::endl;

  // std::string strHoliday (hoilday);
  // strHoliday = Date(3, 25, 2021);
  // 注意以上这样赋值导致隐式转换，
  // 即为了让赋值通过编译器而不引发错误，编译器使用了可用的转换运算符
  // 为了禁止隐式转换，在运算符声明开头使用关键字 explicit ，
  // 要求程序员使用强制类型转换来确认转换意图
  std::cout << "hoilday is on: " << (const char*)hoilday << std::endl;
  
  return 0;
}

// $ g++ -o main 12.2_conversion_operator.cpp 
// $ ./main.exe 
// hoilday is on: 3 / 24 / 2021