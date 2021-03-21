## Assignment 作业
- 作业包括测验和练习，前者帮助读者加深对所学知识的理解，后者提供了使用新学知识的机会。
- 请尽量先完成测验和练习题，继续学习下一章前，请务必弄懂这些答案。

### 10.9.1 测验
1. 我希望基类的某些成员可以在派生类中访问，但不能在继承层次结构外访问，该使用那种访问限定符？
- protected 访问限定符

2. 如果一个函数接受一个基类对象作为参数，而我将一个派生类对象作为实参按值传递给它，结果将如何？
- 切除行为的结果无法预计

3. 该使用私有继承还是组合？
- 使用组合，提高设计的灵活性
- 除非必要，不建议使用 private 继承

4. 在继承层次结构中，关键字 using 有何用途？
- using 可以用于解除基类方法的隐藏 编译错误

5. Derived 类以私有方式继承了 base 类，而 SubDerived 类以公有方式继承了 Derived 类。请问 SubDerived 类能够访问 Base 类的公有成员吗？
- 不能


### 10.9.2 练习
1. 创建程序清单 10.10 所示的 Platypus 对象时，将以什么样的顺序调用构造函数？
- 构造顺序与类声明中指定的顺序相同，依次 Mammal, Reptile, Bird, Platypus
- 析构顺序则是相反的(与构造顺序）

2. 使用代码说明 Polygon、Triangle 和 Shape 类之间的关系？
```C++
class Shape {};
class Polygon: public Shape {};
class Triangle: public Polygon {};
```
- Triangle 继承于 Polygon
- Polygon 继承于 Shape

3. D2 类继承了 D1  类，而 D1 类继承了 Base 类。要禁止 D2 类访问 Base 的公有方法，应该使用那种访问限定符？在什么地方使用？
- D1 类和 Base 类之间的继承关系为私有继承

```C++
class Base {};
class D1: private Base {};
class D2: public Base {};
```

4. 编写下面的代码表示哪种继承关系？
```C++
class Derived: Base
{
// ... Derived members
};
```

- 类的继承关系默认为私有
- 结构体的继承关系默认为公有

5. 查错：下述代码有何问题？
```C++
class Derived: public Base
{
// ... Derived members
};
void SomeFunc (Base value)
{
// …
}
```

- 函数按值传递 Base 类型的参数，编译器进行二进制复制时，产生了切除行为
- 将导致不可预测和不稳定的结果