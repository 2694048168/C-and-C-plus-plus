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
    if (initString != NULL)
    {
      // C-style string '\0'
      buffer = new char[strlen(initString) + 1];
      strcpy(buffer, initString);
    }
    else
    {
      buffer = NULL;
      std::cout << "The initString is illegal." << std::endl;
    }
  }

  // deconstructor
  // 字符串类中使用了析构函数的特性，再更智能使用指针方面，析构函数扮演重要角色
  ~MyString()
  {
    std::cout << "Invoking destructor, clearing up." << std::endl;
    // if (buffer != NULL)
    if (!buffer)
    {
      delete [] buffer;
    }
  }

  const char* GetString()
  {
    return buffer;
  }
};


int main(int argc, char** argv)
{
  MyString sayHello("Hello from String Class");
  std::cout << "String buffer in sayHello is " << sayHello.GetString();
  std::cout << "characters long" << std::endl;
  std::cout << "Buffer contains: " << sayHello.GetString() << std::endl;

  return 0;
}

// $ g++ -o main 9.7_deconstructor.cpp 
// $ ./main.exe 
// String buffer in sayHello is Hello from String Classcharacters long
// Buffer contains: Hello from String Class
// Invoking destructor, clearing up.