## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 26.7.1 测验
1. 为应用程序编写自己的智能指针应该查看什么地方？
- www.boost.org
- 注意实现复制和赋值功能

2. 智能指针是否会严重降低应用程序的性能？
- 不会
- 前提而言是选择正确并且编写好的智能指针

3. 引用计算智能指针在什么地方储存引用计算？
- 如果是入侵式，将由指针拥有的对象保存引用计数；
- 否则，指针将这种信息保存在自由储存区的共享对象中

4. 引用链接指针使用的链接表机制是单向链表还是双向链表？
- 需要双向遍历链链表，故此需要双向链表


### 26.7.2 练习
1. 查错：指出下述代码中的错误：

```C++
std::auto_ptr<SampleClass> object (new SampleClass ());
std::auto_ptr<SampleClass> anotherObject (object);
object->DoSomething ();
anotherObject->DoSomething();
```

- object->DoSomething (); 错误
- 指针在复制时候失去了对对象的拥有权，导致非法无效

2. 使用 unique_ptr 类实例化一个 Carp 对象，而 Carp 类继承了 Fish 类。将该对象作为 Fish 指针传递时是否会出现切除问题。
- 不会
- 以引用方式进行，不需要复制，不会出现切除问题

```C++
#include <memory>
#include <iostream>
using namespace std;

class Fish
{
public:
  Fish() { cout << "Fish: Constructed!" << endl; }
  ~Fish() { cout << "Fish: Destructed!" << endl; }
  void Swim() const { cout << "Fish swims in water" << endl; }
};

class Carp : public Fish
{
};

void MakeFishSwim(const unique_ptr<Fish> &inFish)
{
  inFish->Swim();
}

int main(int argc, char **argv)
{
  unique_ptr<Fish> myCarp(new Carp); // note this
  MakeFishSwim(myCarp);
  return 0;
}
```

3. 查错：指出下述代码中的错误：

```C++
std::unique_ptr<Tuna> myTuna (new Tuna);
unique_ptr<Tuna> copyTuna;
copyTuna = myTuna;
```

- std::unique_ptr 的复制构造函数和赋值运算符都是私有的
- 不允许赋值和复制