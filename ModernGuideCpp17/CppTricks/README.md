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

**静态多态和动态多态对比**

| 特性       | CRTP（静态多态）          | 动态多态 |
|  ----     | ----          |----  |
| 性能	    | 高效，无运行时开销	| 有虚函数表查找开销，性能略低 |
| 灵活性	    | 受限，类型在编译时固定	| 灵活，类型可以在运行时动态选择 |
| 类型安全性	    | 高，编译时检查	| 低，存在类型转换失败风险 |
| 编译期 vs 运行期	    | 完全在编译时	| 依赖运行时 |
| 耦合性	    | 较高，A 模块使用 B 模块中的 CRTP 实现，涉及到的符号都得对 A 模块可见	| 较低，A 模块使用 B 模块中的接口类，接口实际实现的类不需要对 A 模块暴露 |
| 可读性	    | 很差，涉及到模版，还存在代码体积膨胀问题	| 较差 |

