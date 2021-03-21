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
    if (initString != NULL)
    {
      // C-style string '\0'
      buffer = new char[strlen(initString) + 1];
      strcpy(buffer, initString);

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
  // 编译器对于整型、字符和原始指针等 POD数据 执行二进制复制拷贝
  // 二进制复制拷贝不复制指向的内存地址
  // 故此会导致访问一块已经被销毁的内存地址，引起非法访问错误
  // 程序结果两次调用析构函数对同一内存地址进行操作！！！
  UseMyString(sayHello);

  return 0;
}

// $ g++ -o main 9.8_copy_constructor.cpp 
// $ ./main.exe 

// String buffer in MyString is 23 characters long
// buffer contains: Hello from String Class       
// Invoking destructor, clearing up.
// Invoking destructor, clearing up.