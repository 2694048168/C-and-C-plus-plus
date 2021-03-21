#include <iostream>

/**3. 禁止再栈中实例化的类
 * 栈空间通常是有限的资源
 * 特别是在对数据库等进行操作时，禁止在栈上实例化，应该在自由存储区创建其实例对象
 * 关键的实现 将析构函数声明为私有的
 */
class MonsterDB
{
private:
  // private destructor to prevent instances on stack
  ~MonsterDB() {};

public:
  // 提供一个销毁实例的静态共有函数
  static void DestroyInstance(MonsterDB* ptrInstance)
  {
    // member can invoke private destructor
    delete ptrInstance;
  }

  void DoSomething()
  {
    std::cout << "Doing something work" << std::endl;
  }
};

int main(int argc, char** argv)
{
  // create instance of class on heap, not on stack.
  MonsterDB* myDB = new MonsterDB();
  myDB->DoSomething();

  // 下面注释代码不能通过编译器，实现禁止在栈上创建实例
  // uncomment next line to see compile failure
  // delete myDB; // private destructor cannot be invoked

  // using static member to release memory.
  MonsterDB::DestroyInstance(myDB);

  return 0;
}