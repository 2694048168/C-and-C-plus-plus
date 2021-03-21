## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 24.7.1 测验
1. 能否修改 priority_queue 的行为，使得值最大的元素最后弹出？
- 可以
- 使用一个二元谓词进行判断，对优先队列里面的元素进行排序 

2. 假设有一个包含 Coin 对象的 priority_queue，要让 priority_queue 将币值最大的硬币放在队首，需要为 Coin 定义那种运算符？
- 运算符 <

3. 假设有一个包含 6 个 Coin 对象的 stack，能否访问或者删除第一个插入的 Coin 对象？
- 可以
- 将栈顶的元素逐一弹出，直到最后一个就可以访问或者删除第一个插入的 Coin 对象


### 24.7.2 练习
1. 邮局有一个包含人 (Person 类) 的队列。Person 包含两个成员属性，分别用于储存年龄和性别，其定义如下，请改进这个类，使得包含其对象的 priority_queue 优先向老人和妇女提供服务。
- 提供一个二元谓词即可，对运算符 < 进行重载即可

```C++
class Person
{
public:
  int age;
  bool isFemale;
};
```

```C++
class Person
{
public:
  int age;
  bool isFemale;

  bool operator < (const Person& anotherPerson) const
  {
    bool bRet = false;
    if (age > anotherPerson.age)
    {
      bRet = true;
    }
    else if (isFemale && anotherPerson.isFemale)
    {
      bRet = true;
    }

    return bRet;
  }
};
```

2. 编写一个程序，使用 stack 类反转用户输入的字符串的排列顺序。
- 参考文件 24.5_test_stack.cpp

```C++
#include <iostream>
#include <stack>

int main(int argc, char** argv)
{
  std::cout << "Please enter 5 words what you want: ";
  std::stack<std::string> strInStack;
  for (size_t i = 0; i < 5; ++i)
  {
    std::string str;
    std::cin >> str;
    strInStack.push(str);
  }

  std::cout << "--------------------------------" << std::endl;
  std::cout << "The reverse string: " << std::endl;
  while (!strInStack.empty())
  {
    std::cout << strInStack.top() << ' ';
    strInStack.pop();
  }
  
  return 0;
}

// $ g++ -o main 24.5_test_stack.cpp 
// $ ./main.exe 

// Please enter 5 words what you want: li wei jxufe software hardware
// --------------------------------
// The reverse string:
// hardware software jxufe wei li
```