#include <iostream>

// 转换运算符
class Date
{
public:
  // constructor using list initialization
  Date(int inputMonth, int inputDay, int inputYear)
      : month(inputMonth), day(inputDay), year(inputYear)
  {

  }

  // 使用 explicit 要求使用强制类型转换来确认转换意图
  explicit operator int()
  {
    return (year * 365) + (month * 30) + day;
  }

private:
  int day, month, year;
};

int main(int argc, char** argv)
{
  Date hoilday (3, 24, 2021);
  // 要求程序员使用强制类型转换来确认转换意图
  std::cout << "hoilday is: " << (int)hoilday << " days."<< std::endl;
  
  return 0;
}

// $ g++ -o main 12.10_test_conver.cpp 
// $ ./main.exe 
// hoilday is: 737779 days.