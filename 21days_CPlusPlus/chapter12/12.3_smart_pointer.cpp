#include <iostream>
#include <memory>  // using unique_ptr

class Date
{
public:
  // constructor using list initialization
  Date(int inputMonth, int inputDay, int inputYear)
      : month(inputMonth), day(inputDay), year(inputYear)
  {

  }

  void DisplayDate()
  {
    std::cout << month << " / " << day << " / " << year << std::endl;
  }

private:
  int day, month, year;
};

int main(int argc, char** argv)
{
  // smart pointer 智能指针
  std::unique_ptr<int> smartIntPtr(new int);
  *smartIntPtr = 42;

  // 解引用运算符 * 和成员选择运算符 -> 在智能指针编程中应用很广
  // 智能指针是封装常规指针的类，旨在通过管理所有权和复制问题简化内存管理
  // 有些情况下，智能指针能够提高程序的性能
  // 对 * 和 -> 运算符进行重载，帮助智能指针完成工作

  // using smart pointer type like an int*
  std::cout << "Integer value is: " << *smartIntPtr << std::endl;

  std::unique_ptr<Date> smartHoildayPtr (new Date(4, 2, 2021));
  std::cout << "The new instance of date contains: ";

  // using smartHoilday just as you would a Date*
  smartHoildayPtr->DisplayDate();
  
  return 0;
}

// $ g++ -o main 12.3_smart_pointer.cpp 
// $ ./main.exe 
// Integer value is: 42
// The new instance of date contains: 4 / 2 / 2021