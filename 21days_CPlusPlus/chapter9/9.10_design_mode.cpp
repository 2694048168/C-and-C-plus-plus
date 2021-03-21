/**1.不允许复制的类
 * C++ 提供了实现这种设计模式的解决方案
 * 要禁止类对象被复制，可以声明一个私有的拷贝构造函数
 * 为了禁止赋值，可以声明一个私有的赋值运算符
 */
/*
class Solution
{
private:
  // private copy constructor
  Solution(const Solution&);

  // private copy assignment operator
  Solution& operator= (const Solution&);
public:

};
*/

/**2. 只能有一个实例的单例类
 * 单例设计模式
 * 使用私有构造函数、私有赋值运算符和静态实例成员
 * static 关键字
 */
#include <iostream>
#include <string>

class President
{
private:
  // private default constructor
  President() {};

  // private copy constructor
  President(const President&);
  
  // private assignment operator
  const President& operator= (const President&);

  std::string name;

public:
  // static objects are constructor only once.
  static President& GetInstance()
  {
    static President onlyInstance;
    return onlyInstance;
  }

  std::string GetName()
  {
    return name;
  }

  void SetName(std::string InputName)
  {
    name = InputName;
  }
};


int main(int argc, char** argv)
{
  President& onlyPresident = President::GetInstance();
  onlyPresident.SetName("Abraham Lincoln");

  // 一下各种创建 Presidient 实例和拷贝的方式，都不能编译通过，达到了单例的设计模式
  // uncomment lines to see how compile failures prohibit duplicates
  // President second; // cannot access constructor
  // President* third= new President(); // cannot access constructor
  // President fourth = onlyPresident; // cannot access copy constructor
  // onlyPresident = President::GetInstance(); // cannot access operator=

  std::cout << "The name of the Presidient is: " << President::GetInstance().GetName() << std::endl;

  return 0;
}