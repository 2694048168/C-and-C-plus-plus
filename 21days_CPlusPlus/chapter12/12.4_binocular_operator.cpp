#include <iostream>

// 双目运算符 binocular operator
// 对两个操作数进行操作的运算符
// 以全局函数或者静态成员函数的方式实现的双目运算符定义
// return_type operator_type (parameter1, parameter2);

// 类成员函数方式实现的双目运算符定义
// 只接受一个参数，第二个参数通常从类属性获取
// return_type operator_type (parameter1);

class Date
{
public:
  // using list initianization for constructor.
  Date(int inputMonth, int inputDay, int inputYear) 
      : month(inputMonth), day(inputDay), year(inputYear)
      {};

  // 针对类实现对双目加法运算符进行重载
  // binary addition
  Date operator + (int daysToAdd)
  {
    Date newDate(month, day + daysToAdd, year);
    return newDate;
  }

  // 针对类实现 复合运算符 += 进行重载
  void operator += (int daysToAdd)
  {
    day += daysToAdd;
  }

  // 针对类实现对双目减法运算符进行重载
  // binary substraction
  Date operator - (int daysToSub)
  {
    return Date(month, day - daysToSub, year);
  }

  // 针对类实现 复合运算符 -= 进行重载
  void operator -= (int daysToSub)
  {
    day -= daysToSub;
  }

  // 针对类实现 == 运算符进行重载
  bool operator == (const Date& compareTo)
  {
    return ((day == compareTo.day)
            && (month == compareTo.month)
            && (year == compareTo.year));
  }

  // 针对类实现 != 运算符进行重载
  bool operator != (const Date& compareTo)
  {
    return !(this->operator==(compareTo));
  }

  // 针对类实现 < 运算符进行重载
  bool operator < (const Date& compareTo)
  {
    if (year < compareTo.year)
    {
      return true;
    }
    else if (month < compareTo.month)
    {
      return true;
    }
    else if (day < compareTo.day)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  // 针对类实现 <= 运算符进行重载
  bool operator <= (const Date& compareTo)
  {
    if (this->operator== (compareTo))
    {
      return true;
    }
    else
    {
      return this->operator< (compareTo);
    }
  }

  // 针对类实现 > 运算符进行重载
  bool operator > (const Date& compareTo)
  {
    return !(this->operator<= (compareTo));
  }

  // 针对类实现 >= 运算符进行重载
  bool operator >= (const Date& compareTo)
  {
    if (this->operator== (compareTo))
    {
      return true;
    }
    else
    {
      return this->operator> (compareTo);
    }
  }

  // test
  void DisplayDate()
  {
    std::cout << month << " / " << day << " / " << year << std::endl;
  }

private:
  int day, month, year;
  std::string dateInString;
};

int main(int argc, char** argv)
{
  // test binocular operator for Class Date.
  Date holiday(23, 9, 2021);
  std::cout << "holiday on: ";
  holiday.DisplayDate();

  // test - operator for class Date.
  std::cout << "===========================" << std::endl;
  Date previousHoliday(holiday - 6);
  std::cout << "Previous Holiday on: ";
  previousHoliday.DisplayDate();

  // test -= operator for class Date.
  std::cout << "===========================" << std::endl;
  std::cout << "holiday -= 5 give: ";
  holiday -= 5;
  holiday.DisplayDate();

  // test + operator for class Date.
  std::cout << "===========================" << std::endl;
  Date nextHoliday(holiday + 3);
  std::cout << "Next Holiday on: ";
  nextHoliday.DisplayDate();

  // test += operator for class Date.
  std::cout << "===========================" << std::endl;
  std::cout << "holiday += 4 give: ";
  holiday += 5;
  holiday.DisplayDate();

  // test == operator for class Date.
  std::cout << "===========================" << std::endl;
  if (holiday == nextHoliday)
  {
    std::cout << "Equality operator: The two are on the same day." << std::endl;
  }
  else
  {
    std::cout << "Equality operator: The two are on the different day." << std::endl;
  }
  
  // test == operator for class Date.
  std::cout << "===========================" << std::endl;
  if (holiday != previousHoliday)
  {
    std::cout << "Equality operator: The two are on the different day." << std::endl;
  }
  else
  {
    std::cout << "Equality operator: The two are on the same day." << std::endl;
  }

  // test < operator for class Date.
  std::cout << "===========================" << std::endl;
  if (holiday < previousHoliday)
  {
    std::cout << "operator<: holiday happens first." << std::endl;
  }

  // test > operator for class Date.
  std::cout << "===========================" << std::endl;
  if (holiday > previousHoliday)
  {
    std::cout << "operator>: holiday happens later." << std::endl;
  }

  // test <= operator for class Date.
  std::cout << "===========================" << std::endl;
  if (holiday <= previousHoliday)
  {
    std::cout << "operator<=: holiday happens on or before previousHoliday." << std::endl;
  }

  // test >= operator for class Date.
  std::cout << "===========================" << std::endl;
  if (holiday >= previousHoliday)
  {
    std::cout << "operator>=: holiday happens on or after previousHoliday." << std::endl;
  }
  
  return 0;
}

// $ g++ -o main 12.4_binocular_operator.cpp 
// $ ./main.exe

// holiday on: 23 / 9 / 2021
// ===========================
// Previous Holiday on: 23 / 3 / 2021
// ===========================
// holiday -= 5 give: 23 / 4 / 2021
// ===========================
// Next Holiday on: 23 / 7 / 2021
// ===========================
// holiday += 4 give: 23 / 9 / 2021
// ===========================
// Equality operator: The two are on the different day.     
// ===========================
// Equality operator: The two are on the different day.     
// ===========================
// ===========================
// operator>: holiday happens later.
// ===========================
// ===========================
// operator>=: holiday happens on or after previousHoliday.