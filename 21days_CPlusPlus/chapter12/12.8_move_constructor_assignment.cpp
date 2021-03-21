#include <iostream>
#include <cstring>

/**用于高性能编程的移动构造函数和移动赋值运算符
 * std=C++11 标准的一部分，提高性能，旨在避免复制不必要的临时值
 * 如果提供了移动构造函数和移动赋值运算符，
 * 编译器意识到需要创建临时拷贝时候调用，
 * 这对于管理动态分配资源的类十分重要和高效
 * 在需要创建临时右值，C++11的编译器将使用移动构造函数和移动赋值运算符，
 * 在两者的实现中，只是将资源从 源 移动到 目的地，而没有进行复制
 */

class MyString
{
public:
  // overload constructor.
  MyString(const char* initialInput)
  {
    std::cout << "Constructor called for: " << initialInput << std::endl;
    // if (initialInput != NULL)
    if (!initialInput)
    {
      buffer = new char [strlen(initialInput) + 1];
      strcpy(buffer, initialInput);
    }
    else
    {
      buffer = NULL;
    }
  }

  // move constructor.
  MyString(MyString&& moveSrc)
  {
    std::cout << "Move constructor moves: " << moveSrc.buffer << std::endl;
    if (moveSrc.buffer != NULL)
    {
      buffer = moveSrc.buffer;  // take ownershiop i.e. 'move'
      moveSrc.buffer = NULL; // free move source
    }
  }

  // move assignment operator
  MyString& operator = (MyString&& moveSrc)
  {
    std::cout << "Move assignment operator moves: " << moveSrc.buffer << std::endl;
    if ((moveSrc.buffer != NULL) && (this != &moveSrc))
    {
      delete [] buffer; // release own buffer

      buffer = moveSrc.buffer;  // take ownershiop i.e. 'move'
      moveSrc.buffer = NULL;  // free move source
    }

    return *this;
  }

  // copy constructor
  MyString(const MyString& copySrc)
  {
    std::cout << "Copy constructor copies: " << copySrc.buffer << std::endl;
    if (copySrc.buffer != NULL)
    {
      buffer = new char[strlen(copySrc.buffer) + 1];
      strcpy(buffer, copySrc.buffer);
    }
    else
    {
      buffer = NULL;
    }
  }

  // copy assignment operator.
  MyString& operator= (const MyString& copySrc)
  {
    std::cout << "Copy assignment operator copies: " << copySrc.buffer << std::endl;
    if ((this != &copySrc) && (copySrc.buffer != NULL))
    {
      if (buffer != NULL)
      {
        delete [] buffer;
      }
      buffer = new char[strlen(copySrc.buffer) + 1];
      strcpy(buffer, copySrc.buffer);
    }

    return *this;
  }

  // destructor
  ~MyString()
  {
    // if (buffer != NULL)
    if (!buffer)
    {
      delete [] buffer;
    }
  }

  int GetLength()
  {
    return strlen(buffer);
  }

  operator const char*()
  {
    return buffer;
  }

  // operator + overloading
  MyString operator + (const MyString& addThis)
  {
    std::cout << "operator+ called: " << std::endl;
    MyString newStr;

    if (addThis.buffer != NULL)
    {
      newStr.buffer = new char[GetLength() + strlen(addThis.buffer) + 1];
      strcpy(newStr.buffer, buffer);
      strcpy(newStr.buffer, addThis.buffer);
    }

    return newStr;
  }

private:
  char * buffer;

  // private default constructor.
  MyString() : buffer(NULL)
  {
    std::cout << "Default constructor called." << std::endl;
  }
};

// 移动构造函数和移动赋值函数的实现，
// 移动的基本语义基本上是通过接管移动源中的资源的所有权实现的
// 避免不必要的复制和内存分配，节省处理时间，提高性能
int main(int argc, char** argv)
{
  MyString hello("Hello ");
  MyString world(" world");
  MyString cpp(" of C-Plus-Plus.");

  MyString sayHelloAgain("overwrite this");
  sayHelloAgain = hello + world + cpp;
  
  return 0;
}

// $ g++ -o main 12.8_move_constructor_assignment.cpp 
// $ ./main.exe 

// Constructor called for: Hello
// Constructor called for:  world
// Constructor called for:  of C-Plus-Plus.
// Constructor called for: overwrite this  
// operator+ called:
// Default constructor called.
// operator+ called:
// Default constructor called.
// Move assignment operator moves:
