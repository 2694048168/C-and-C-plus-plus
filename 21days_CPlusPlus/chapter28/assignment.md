## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 28.8.1 测验
1. std::exception 是什么？
- C++ 异常基类

2. 使用 new 分配内存失败时，将引发那种异常？
- 引发异常类型：std::bad_alloc

3. 在异常处理程序中(catch 块)，为大量 int 变量分配内存以便备份数据合适吗？
- 不合适

4. 假设有一个异常类 MyException, 它继承 std::exception，将如何捕获这种异常对象？
- 异常是一个对象，对 std::exception 的 虚函数 what 重写，可以自定义任何信息


### 28.8.2 练习
1. 查错：指出下述代码中的错误：

```C++
class SomeIntelligentStuff
{
bool isStuffGoneBad;
public:
  ~SomeIntelligentStuff()
  {
    if(isStuffGoneBad)
      throw "Big problem in this class, just FYI";
  }
};
```

- 析构函数引发异常，会导致程序终止

2. 查错：指出下述代码中的错误：

```C++
int main()
{
  int* millionNums = new int [1000000];
  // do something with the million integers
  delete []millionNums;
}
```

- 没有异常处理机制，new 申请内存失败，将引发未知错误

3. 查错：指出下述代码中的错误：

```C++
int main()
{
  try
  {
    int* millionNums = new int [1000000];
    // do something with the million integers
    delete []millionNums;
  }
  catch(exception& exp)
  {
    int* anotherMillion = new int [1000000];
    // take back up of millionNums and save it to disk
  }
}
```

- 在 cartch block 中引发异常，建议不要这样操作
- 如果 try block 也引发异常，那么将导致恶性循环