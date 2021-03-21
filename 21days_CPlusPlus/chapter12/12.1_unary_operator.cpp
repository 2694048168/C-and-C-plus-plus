#include <iostream>

/**运算符类型和运算符重载
 * 类能够封装运算符，简化对实例的执行操作
 * 与函数一样，运算符也可以重载
 * 从语法层面看，运算符与函数没啥区别，多了关键字 operator
 * return_type operator operator_symbol (parameter list);
 * 
 * 对类的实现相关运算符，需要额外的工作，但是类使用起来更加容易
 * 根据操作数，运算符可以划分为单目运算符(unary operator)和多目运算符(multi order operator)
 * 
 * // 实现全局函数或者静态成员函数的单目运算符
 * return_type operator operator_symbol (parameter list)
 * {
 *    // implementation
 * }
 * // 类成员(非静态函数)的单目运算符没有参数，唯一参数就是当前类实例 (*this)
 * return_type operator operator_symbol ()
 * {
 *    // implementation
 * }
 * 
 * 可以重载的单目运算符 ++ -- + - * / % * ~ & ！-> 
 */

// 单目递增++ 单目递减-- 运算符
class Date
{
public:
  Date(int inputMonth, int inputDay, int inputYear)
      : month(inputMonth), day(inputDay), year(inputYear) {};

  // prefix increment
  Date& operator ++ ()
  {
    ++day;
    return *this;
  }

  // postfix increment
  Date operator ++ (int)
  {
    Date copy(month, day, year);
    ++day;
    // copy of instance before increment returned
    return copy;
  }

  // prefix decrement
  Date& operator -- ()
  {
    --day;
    return *this;
  }

  // postfix decrement
  Date operator -- (int)
  {
    Date copy(month, day, year);
    --day;
    // copy of instance before increment returned
    return copy;
  }

  void DisplayDate()
  {
    std::cout << month << " / " << day << " / " << year << std::endl;
  }

private:
  int month;
  int day;
  int year;
};

int main(int argc, char** argv)
{
  Date holiday(3, 24, 2021);
  std::cout << "The date object is initialized to: ";
  holiday.DisplayDate();

  // ++holiday prefix increment
  ++holiday;
  std::cout << "Date after prefix-increment is: ";
  holiday.DisplayDate();

  // --holiday prefix decrement
  --holiday;
  std::cout << "Date after prefix-decrement is: ";
  holiday.DisplayDate();

  std::cout << "===========================" << std::endl;
  // 创建一个未使用的临时拷贝 postfix

  // holiday++ postfix increment
  holiday++;
  std::cout << "Date after postfix-increment is: ";
  holiday.DisplayDate();

  // holiday-- prefix decrement
  holiday--;
  std::cout << "Date after postfix-decrement is: ";
  holiday.DisplayDate();
  
  return 0;
}


// $ g++ -o main 12.1_unary_operator.cpp 
// $ ./main.exe 
// The date object is initialized to: 3 / 24 / 2021
// Date after prefix-increment is: 3 / 25 / 2021   
// Date after prefix-decrement is: 3 / 24 / 2021   
// ===========================
// Date after postfix-increment is: 3 / 25 / 2021  
// Date after postfix-decrement is: 3 / 24 / 2021 