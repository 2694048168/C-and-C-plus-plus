## Modern C++ tricks

> the Best Practices tricks for Modern C++

> What I cannot create, I do not understand. Know how to solve every problem that has been solved. Quote from Richard Feynman


![memory-order](https://i-blog.csdnimg.cn/blog_migrate/e84a9e16b0b605659b5b81a3110f402d.png)

```shell
cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE:STRING=Debug 
cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_BUILD_TYPE:STRING=Release 

cmake --build build --config Debug
cmake --build build --config Release
```

### C++设计函数内部实现的建议

> 函数是 C++ 代码实现功能的具体表示, 所以优化代码的结构、性能和可维护性对写出高质量的c++代码极为重要

- 确保参数有效性: 
    - 函数入口处的参数检查非常重要，可以确保输入是有效的，避免后续代码执行时出错
    - 对指针类型的参数进行空指针检查
    - 对整数、浮点数等类型的参数进行范围检查
    - 对容器进行空检查
- 减少重复代码(DRY 原则): 
    - 如果函数内部出现重复代码,可以将重复部分提取为辅助函数,提升代码可读性和可维护性
    - inline small helper-function
    - lambda expression function
- 处理异常: 
    - 在函数内部进行异常捕获和处理, 确保函数在运行过程中遇到问题时不会导致程序崩溃; 
    - 对于可能会引发异常的代码块, 使用 try-catch 进行管理;
    - 建议不使用 try-catch 模式
    - 采用 Result Status 返回 error-no 枚举, 外部调用者做处理
- 局部变量初始化: 
    - 始终确保局部变量在使用前被初始化, 以避免未定义行为; 尤其是在类成员函数中, 可能有未初始化的成员变量
    - POD 数据一定声明即初始化
    - 结构体在声明并给定合理初始值
    - 指针声明即初始化 nullptr
    - 类变量声明即列表初始化 {}
    - 采用 memset(Struct_Variable, 0, sizeof(Struct))
- 合理使用早退出(Guard Clauses): 
    - 使用早退出（guard clauses）来处理错误或特殊情况, 避免嵌套代码, 提高代码的可读性
- 优化循环结构:
    - 在循环中如果能减少不必要的操作或计算, 就能显著提高性能; 例如将不变的表达式移到循环外
    - 现代编译器开启优化, 编译器会对**循环**部分代码做优化
- 明确函数的职责: 
    - 一个函数应尽量只做一件事, 遵循单一职责原则(SRP)
    - 如果函数内部的逻辑过于复杂, 考虑将其拆分成多个小函数, 以提高可读性和维护性
- 减少嵌套:
    - 函数内部嵌套过多会影响可读性, 尽量通过逻辑提取或早退出来减少嵌套层次
    - 不要出现 if-if-if-if 多层嵌套的情况


### CRTP（Curiously Recurring Template Pattern）是一种 C++ 编程技巧

> CRTP(Curiously Recurring Template Pattern)是一种 C++ 编程技巧, 使用模板类和继承的组合来实现静态多态. 该模式的关键思想是: 在模板类的定义中, 模板参数是当前类自身(通常是派生类), 这个技巧通常用于实现编译时多态, 优化性能. 因为C++运行时多态,需要虚函数和虚函数表的支持, 存在一定的性能开销.

```C++
// 先定义一个模板类作为基类
template <typename T>
class Base
{
    // ...
};

 // 定义一个派生类，这个类继承以自身作为参数的基类
class Derived : public Base<Derived>
{
    // ...
};
```

#### **静态多态(CRTP)和动态多态对比**

| 特性       | CRTP（静态多态）          | 动态多态 |
|  ----     | ----          |----  |
| 性能	    | 高效，无运行时开销	| 有虚函数表查找开销，性能略低 |
| 灵活性	    | 受限，类型在编译时固定	| 灵活，类型可以在运行时动态选择 |
| 类型安全性	    | 高，编译时检查	| 低，存在类型转换失败风险 |
| 编译期 vs 运行期	    | 完全在编译时	| 依赖运行时 |
| 耦合性	    | 较高，A 模块使用 B 模块中的 CRTP 实现，涉及到的符号都得对 A 模块可见	| 较低，A 模块使用 B 模块中的接口类，接口实际实现的类不需要对 A 模块暴露 |
| 可读性	    | 很差，涉及到模版，还存在代码体积膨胀问题	| 较差 |

CRTP 能在编译时就规划好所有的函数调用路径,特别适合那些需要高性能,同时又想要优雅地复用代码的场景:
- 游戏引擎中的组件系统
- 高性能计算库
- 图形渲染管线

CRTP 的"安全检查员"是怎么工作,保证编译期类型正确
- C++ 的魔法,用编译时多态的方式,代码既快速又优雅
- C++ 的魔法, 静态检查不是限制,而是保护,它让代码更安全、更可靠
- CRTP 和链式调用的完美组合

```C++
template<typename Derived>
class Shape {
public:
    // 这位可爱的检查员会确保所有图形都乖乖继承自 Shape 哦！🎯
    static_assert(std::is_base_of<Shape<Derived>, Derived>::value,
                 "哎呀呀，你是不是忘记继承 Shape 啦？快去补救吧！🤔");
                 
    double area() {
        // 这里的检查就像是点名一样，确保每个图形都会计算自己的面积 📏
        static_assert(std::is_member_function_pointer<
            decltype(&Derived::computeArea)>::value,
            "咦？computeArea 方法不见啦！是不是忘记写啦？✍️");
            
        // 通过检查的小可爱就可以愉快地计算面积啦～ 🌟
        return static_cast<Derived*>(this)->computeArea();
    }
};
```

#### 性能对比：CRTP vs 虚函数, 为什么 CRTP 能比虚函数快
- 虚函数的工作方式:
    - 每个带虚函数的类都有一个虚函数表(vtable)
    - 每次调用虚函数时都需要:
    - 这些间接操作会带来性能开销
        1. 查找对象的 vtable 指针
        2. 在 vtable 中找到正确的函数地址
        3. 通过函数指针进行调用
- CRTP 的工作方式:
    - 在编译时就确定了所有函数调用
    - 编译器可以直接内联函数调用
    - 没有运行时查表开销
    - 不需要存储额外的 vtable 指针

> 虚函数就像是在跑步时需要不断查看路标的选手, 而 CRTP 就像是把整个路线图都记在脑子里的选手, 当然要跑得更快啦. 小贴士: 在现代 CPU 中, **间接跳转**(比如虚函数调用)可能会导致**分支预测失败**, 进一步影响性能, 而 CRTP 的直接调用则完全避免了这个问题.

#### CRTP 的局限性：there's no such thing as a free lunch
- 编译时绑定的限制,无法像虚函数那样灵活地进行运行时多态
- CRTP 的接口变更的烦恼
    - 模板的特性决定了所有使用这个基类的代码都需要看到完整的定义
        - 不像普通类可以只提供声明
        - 模板必须在头文件中完整定义
    - 连锁反应超级可怕
        - 修改基类 -> 所有派生类受影响
        - 派生类变化 -> 使用派生类的代码要重新编译
        - 最后可能整个项目都要重新编译
    - 提前规划好接口
        - 仔细思考可能需要的所有功能,先把蓝图设计好
        - 一次性把接口设计完整,考虑未来可能的扩展
    - 使用组合而不是继承,善用组合来降低耦合度
- 代码膨胀问题
    - 把共同的大块代码放到非模板基类中
    - 使用策略模式分离可复用的代码
- 调试起来有点累
    - 添加静态断言来提供更友好的错误信息,添加适当的检查和注释
    - 使用更清晰的命名约定,使用好的命名规范
    - 添加详细的注释说明,保持代码结构清晰
- 运行时类型检查不太方便
    - 编译时类型检查
    - 自定义类型检查方法

```C++
// =========把共同的大块代码放到非模板基类中
class CommonBase {
protected:
    void heavyOperation() {
        // 把占空间的代码放这里
        // 所有派生类共用这一份! 🎉
    }
};

template<typename Derived>
class Base : protected CommonBase {
    // 这里只放必要的 CRTP 相关代码 ✨
};

// ========使用策略模式分离可复用的代码
class Strategy {
public:
    void complexOperation() {
        // 把复杂操作集中在这里管理 🎮
    }
};

template<typename Derived>
class Base {
    Strategy strategy;  // 通过组合来复用代码 🤝
};

// ========自定义类型检查方法
template<typename Derived>
class Animal {
protected:
    // 给每种动物一个独特的标识 🏷️
    enum class AnimalType { Dog, Cat, Bird };
    
    // 让派生类告诉我们它是什么动物
    virtual AnimalType getType() const = 0;
};

class Dog : public Animal<Dog> {
protected:
    AnimalType getType() const override {
        return AnimalType::Dog;  // 我是汪星人! 🐕
    }
};
```

> 小贴士: 如果项目经常需要修改接口,那么传统的虚函数可能更适合哦! 毕竟灵活性有时候比性能更重要. 在使用 CRTP 时,要像个精明的收纳师一样,把代码合理安排,避免不必要的重复. 在开发 CRTP 代码时,建议先写好单元测试,这样可以更早地发现潜在问题,省得到时候debug到头秃. 如果程序真的需要频繁的运行时类型检查,那么虚函数可能是更好的选择哦！每个工具都有自己的用武之地.

CRTP 最适合这些场景:
- 追求极致性能的应用
- 在编译时就能确定所有类型关系的情况
- 不需要运行时改变对象类型的场景

虚函数更适合:
- 需要运行时多态的场景
- 要通过基类指针/引用操作对象的情况
- 插件式架构或需要动态加载的系统

> 记住啦 在编程世界里没有最好的方案 只有最适合的选择！要权衡性能、灵活性和维护性这些因素,选择最适合的方案

使用 CRTP 的时候要注意以下几点
- 派生类必须正确继承基类模板
- 要小心循环依赖
- 模板代码可能会导致代码膨胀
- 编译错误信息可能比较难懂

**多线程处理图像**

![Concurrency Parallelism](./images/ConcurrencyParallelism.png)

**创建线程的五种类型**
- 使用 std::thread via modern C++ since C++11
- 使用 std::async: 基于 std::thread 的封装,不仅创建了一个线程,还返回一个 std::future 对象,可以用来获取异步操作的结果
- 使用 POSIX 线程 pthread, 尤其是在需要更底层控制时(UNIX-like 系统中使用的标准线程库)
- 使用 Windows 线程 CreateThread, 操作系统级别的线程创建方法
- 使用线程池,管理一组工作线程,允许提交任务给线程池处理,而不是每次都创建和销毁线程,减少资源消耗和提高效率

```C++
#include <iostream>
#include <pthread.h>

void* hello(void*) {
   std::cout << "Hello from pthread\n";
   return nullptr;
}

int main() {
   pthread_t tid;
   pthread_create(&tid, nullptr, hello, nullptr);
   pthread_join(tid, nullptr);
   return 0;
}
```
