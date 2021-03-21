/** this 指针
 * this 包含当前对象的地址 即 void* this = &object;
 * 调用成员方法时，编译器隐式传递 this 指针(即函数调用中不可见的参数)
 * 从编程角度看，this 用途不多，而且大多情况下都是可选的
 * 调用静态方法时，不会隐式传递 this 指针
 * 因为静态函数不与类的实例相关联，由所有实例共享
 * 故此静态函数中使用 this 指针需要显示设置
 */

/** 结构体 & 类
 * 区别：编译器默认的访问限定符不同
 * 结构体默认访问限定符为 public
 * 类默认访问限定符为 private
 * 
 * 区别：编译器默认的继承方式不同
 * 结构体默认以公有方式继承 public
 * 类默认以私有方式继承 private
 */
/*
struct Human
{
  // default public
  
};
*/

#include <iostream>
#include <cstring>

class MyString
{
private:
  char* buffer;

public:
  // default constructor
  MyString(const char* initString)
  {
    buffer = NULL;
    if (!initString)
    {
      buffer = new char[strlen(initString) + 1];
      strcpy(buffer, initString);
    }
  }

  // copy constructor
  MyString(const MyString& copySource)
  {
    buffer = NULL;
    if (copySource.buffer != NULL)
    {
      buffer = new char[strlen(copySource.buffer) + 1];
      strcpy(buffer, copySource.buffer);
    }
  }

  // default deconstructor
  ~MyString()
  {
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

class Human
{
private:
  int age;
  bool gender;
  MyString name;

public:
  Human(const MyString& inputName, int inputAge, bool gender)
  : name(inputName), age(inputAge), gender(gender)
  {

  }

  int GetAge()
  {
    return age;
  }
};

int main(int argc, char** argv)
{
  MyString manName("Adam");
  MyString womanName("Eve");

  std::cout << "sizeof(MyString) = " << sizeof(MyString) << std::endl;
  std::cout << "sizeof(manName) = " << sizeof(manName) << std::endl;
  std::cout << "sizeof(womanName) = " << sizeof(womanName) << std::endl;

  Human firstMan(manName, 25, true);
  Human firstWoman(womanName, 23, false);

  std::cout << "sizeof(Human) = " << sizeof(Human) << std::endl;
  std::cout << "sizeof(firstMan) = " << sizeof(firstMan) << std::endl;
  std::cout << "sizeof(firstWoman) = " << sizeof(firstWoman) << std::endl;

  return 0;
}


// $ g++ -o main 9.13_sizeof_class.cpp 
// $ ./main.exe 

// sizeof(MyString) = 8   
// sizeof(manName) = 8    
// sizeof(womanName) = 8  
// sizeof(Human) = 16     
// sizeof(firstMan) = 16  
// sizeof(firstWoman) = 16