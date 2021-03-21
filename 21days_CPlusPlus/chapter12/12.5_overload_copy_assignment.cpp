#include <iostream>
#include <cstring>

// 重载复制赋值运算符 = 
// 没有提供 复制赋值运算符，编译器自动添加默认复制赋值运算符，
// 但是并不能完成要求，可能存在类似 “浅拷贝” 问题
/*
classType& operator = (const classType& copySource)
{
  // protection against copy into self
  if (this != &copySource)
  {
    // copy assignment oeprator implementation.
  }
  return *this
}
*/

class MyString
{
private:
  char* buffer;

public:
  // constructor
  MyString(const char* initString)
  {
    // 判断 指针有效性
    buffer = NULL;
    if (initString != NULL)
    {
      // C-style string '\0'
      buffer = new char[strlen(initString) + 1];
      strcpy(buffer, initString);
    }
  }

  // 拷贝构造函数  
  MyString(const MyString& copySource)
  {
    buffer = NULL;
    if (copySource.buffer != NULL)
    {
      // C-style string '\0'
      buffer = new char[strlen(copySource.buffer) + 1];

      // deep copy from the source into local buffer.
      strcpy(buffer, copySource.buffer);
    }
  }

  // deconstructor
  // 字符串类中使用了析构函数的特性，再更智能使用指针方面，析构函数扮演重要角色
  ~MyString()
  {
    // if (buffer != NULL)
    delete [] buffer;
  }

  // 转换运算符，使得类对象可用直接在 std::cout 标准流中直接使用
  operator const char*()
  {
    return buffer;
  }

  // copy assignment operator
  MyString& operator = (const MyString& copySource)
  {
    if ((this != &copySource) && (copySource.buffer != NULL))
    {
      if (buffer != NULL)
      {
        delete [] buffer;
      }
      
      // ensure deep copy by first allcoating own buffer.
      buffer = new char [strlen(copySource.buffer) + 1];

      // copy from the source into local buffer.
      strcpy(buffer, copySource.buffer);
    }

    return *this;
  }
};


int main(int argc, char** argv)
{
  MyString string1 ("hello ");
  MyString string2 (" world");

  std::cout << "Before assignmetn: " << std::endl;
  std::cout << string1 << string2 << std::endl;

  // 赋值运算符
  string2 = string1;
  std::cout << "=========================" << std::endl;
  std::cout << "After assignment string2 = string1: " << std::endl;
  std::cout << string1 << string2 << std::endl;

  return 0;
}

// $ g++ -o main 12.5_overload_copy_assignment.cpp
// $ ./main.exe
// Before assignmetn: 
// hello  world
// =========================
// After assignment string2 = string1: 
// hello hello