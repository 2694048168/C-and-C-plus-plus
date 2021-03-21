#include <iostream>
#include <cstring>

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
  // 拷贝构造函数是一个重载的构造函数
  // 每当对象被复制(拷贝)时，进行深拷贝
  // 有时候对运算符重载也需要进行编写对应运算符的拷贝构造函数
  // 接受一个以 引用 方式传入当前类的对象作为参数，确保对所有缓冲区进行深拷贝

  /** 移动构造函数 move constructor
   * 由于 C++ 特性和需求，有些情况下对象会自动被复制
   * 重复调用拷贝构造函数，同时考虑复制的对象很大
   * 而且只是临时使用对象，那么对性能影响很大
   * 
   * std == C++11 引入 移动构造函数
   * 编译器使用移动构造函数来 “移动” 临时资源，有助于改善性能
   * 移动构造函数通常时利用移动赋值运算符实现的
   */
  // move contructor
  /*
  MyString(MyString&& moveSource)
  {
    if (moveSource.buffer != NULL)
    {
      buffer = moveSource.buffer;  // take ownershiop i.e. 'move' 
      moveSource.buffer = NULL;  // set the move source to NULL
    }
  }
  */
  
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

  // deconstructor
  // 字符串类中使用了析构函数的特性，再更智能使用指针方面，析构函数扮演重要角色
  ~MyString()
  {
    std::cout << "Invoking destructor, clearing up." << std::endl;
    // if (buffer != NULL)
    delete [] buffer;
  }

  int GetLength()
  {
    return strlen(buffer);
  }

  const char* GetString()
  {
    return buffer;
  }
};

void UseMyString(MyString str)
{
  std::cout << "String buffer in MyString is " << str.GetLength() << " characters long" << std::endl;
  std::cout << "buffer contains: " << str.GetString() << std::endl;

  return;
}


int main(int argc, char** argv)
{
  MyString sayHello("Hello from String Class");

  // 使用 UseMyString 函数进行处理，参数按照值传递方式
  // 自动调用 copy constructor 拷贝构造函数，进行深拷贝
  UseMyString(sayHello);

  return 0;
}


// $ g++ -o main 9.9_deep_copy_constructor.cpp
// $ ./main.exe 

// Default constructor: creating new MyString
// buffer points to: 0x0xe06e50
// Copy constructor: copy from MyString
// buffer points to: 0x0xe06e70
// String buffer in MyString is 17 characters long
// buffer contains: Hello from String Class
// Invoking destructor, clearing up.
// Invoking destructor, clearing up.
