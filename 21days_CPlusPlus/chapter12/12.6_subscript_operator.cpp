#include <iostream>
#include <cstring>
#include <string>

// 应该根据类的目标和用途来重载运算符或者实现新的运算符！！！
// 应该根据类的目标和用途来重载运算符或者实现新的运算符！！！
// 应该根据类的目标和用途来重载运算符或者实现新的运算符！！！

// 不能够重载的运算符
// 1. .   成员选择运算符
// 2. .*  指针成员选择运算符
// 3. ::  作用域解析运算符
// 4. ?:  条件三目运算符
// 5. sizeof  获取对象/类 类的大小运算符

class MyString
{
private:
  char* buffer;

  // private default constructor
  MyString() {}

public:
  // constructor
  MyString(const char* initString)
  {
    // 判断 指针有效性
    buffer = NULL;
    std::cout << "Default constructor: creating new MyString" << std::endl;
    if (initString != NULL)
    {
      // C-style string '\0'
      buffer = new char[strlen(initString) + 1];
      strcpy(buffer, initString);

      std::cout << "buffer points to: 0x" << std::hex << (unsigned int*)buffer << std::endl;
    }
  }

  // copy constructor
  MyString(const MyString& copySource)
  {
    buffer = NULL;
    std::cout << "Copy constructor: copy from MyString" << std::endl;
    if (copySource.buffer != NULL)
    {
      // C-style string '\0'
      buffer = new char[strlen(copySource.buffer) + 1];

      // deep copy from the source into local buffer.
      strcpy(buffer, copySource.buffer);

      std::cout << "buffer points to: 0x" << std::hex << (unsigned int*)buffer << std::endl;
    }
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

  // 尽可能使用 const 以免无意间修改数据，并最大限度保护成员属性 
  // 下标运算符 重载
  // const char& operator[] (int index) // use to write / change buffer at index.
  const char& operator[] (int index) const  // used only for accessing char at index.
  {
    if (index < GetLength())
    {
      return buffer[index];
    }
  }

  // deconstructor
  // 字符串类中使用了析构函数的特性，再更智能使用指针方面，析构函数扮演重要角色
  ~MyString()
  {
    if (buffer != NULL)
      delete [] buffer;
  }

  // int GetLength() // 类型匹配
  int GetLength() const
  {
    return strlen(buffer);
  }

  // 转换运算符，使得类对象可用直接在 std::cout 标准流中直接使用
  operator const char*()
  {
    return buffer;
  }
};


int main(int argc, char** argv)
{
  std::cout << "Type a statement: ";
  std::string strInput;
  getline(std::cin, strInput);

  MyString youSaid(strInput.c_str());

  std::cout << "Using operator[] for displaying your input: " << std::endl;
  for (size_t i = 0; i < youSaid.GetLength(); ++i)
  {
    std::cout << youSaid[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Enter index 0 - " << youSaid.GetLength() - 1 << ": ";
  int index = 0;
  std::cin >> index;
  std::cout << "Input character at zero-based position: " 
            << index << " is: " << youSaid[index] << std::endl;

  return 0;
}

// $ g++ -o main 12.6_subscript_operator.cpp 
// 12.6_subscript_operator.cpp: In member function 'const char& MyString::operator[](int) const':
// 12.6_subscript_operator.cpp:75:3: warning: control reaches end of non-void function [-Wreturn-type]
// $ ./main.exe 

// Type a statement: wei li li wei
// Default constructor: creating new MyString
// buffer points to: 0x0x656e60
// Using operator[] for displaying your input:
// w e i   l i   l i   w e i
// Enter index 0 - c: 4
// Input character at zero-based position: 4 is: l