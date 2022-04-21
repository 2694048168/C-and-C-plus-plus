# Effective C++  —— 读书笔记

> Effective C++ 中文版 改善程序与设计的 55 个具体做法 (第三版)

## 前序导读

> 孔子云：“ 取乎其上，得乎其中；取乎其中，得乎其下；取乎其下，则无所得矣 ”。

> 孙中山先生的说法，这个世界依聪明才智的先天高下得三种人：先知先觉得发明家，后知后觉得宣传家，不知不觉得实践家。

**C++ 四种相辅相成的编程范式 (Programming paradigms)**
- Procedural-based Programming 面向过程编程
- Object-based Programming 基于对象编程
- Object-oriented Programming 面向对象编程
- Generics Programming 泛型编程
- (Lambda Programming, 函数式编程)

> [Different of Object-based and Object-oriented](https://stackoverflow.com/questions/15430004/core-difference-between-object-oriented-and-object-based-language)

**C++ 中英文词汇**
- 继承 (inheritance) 
- 面向对象编程 (object-oriented programming)
- 异常 (exceptions)
- 模板 (templates)
- 泛型编程 (generic programming)
- 资源管理 (resource management)
- 模板编程 (programming with templates)
- 单线程系统 (single-threaded systems)
- 多线程系统 (multithreaded systems)
- 用户自定义类型 (user-defined type)

学习程序语言根本大法是一回事；学习如何以某种语言设计并实现高效程序则是另一回事。这种说法对 C++ 尤其适用，因为 C+＋以拥有罕见的威力和丰富的表达能力为傲。

**术语 Terminology**
- 声明式 (declaration) 是告诉编译器某个东西的名称和类型 (type) ，但略去细节。

- 每个函数的声明揭示其签名式 (signature) ，也就是参数和返回类型。一个函数的签名等同于该函数的类型。

- 定义式 (definition) 的任务是提供编译器一些声明式所遗漏的细节。对对象而言，定义式是编译器为此对象拨发内存的地点。对 function 或 function template 而言，定义式提供了代码本体。对 class 或 class template 而言，定义式列出它们的成员。

- 初始化 (Initialization) 是“给予对象初值”的过程。对用户自定义类型的对象而言，初始化由构造函数执行。所谓 default 构造函数是一个可被调用而不带任何实参者。这样的构造函数要不没有参数，要不就是每个参数都有缺省值。

- 构造函数都被声明为 explicit, 这可阻止它们被用来执行隐式类型转换 (implicit type conversions) ，但它们仍可被用来进行显式类型转换 (explicit type conversions)

- copy 构造函数被用来”以同型对象初始化自我对象 ”, copy assignment 操作符被用来“从另一个同型对象中拷贝其值到自我对象”。

- copy 构造函数是一个尤其重要的函数，因为它定义一个对象如何 passed by value （以值传递）。

- STL 是所谓标准模板库 (Standard Template Library), 是 C++ 标准程序库的一部分，容器、迭代器、算法及相关机能。许多相关机能以函数对象 (function objects) 实现，那是“行为像函数”的对象。这样的对象来自于重载 operator() (function call操作符）的 classes。

- 所谓“不明确行为" (undefined behavior)，不明确（未定义）行为的结果是不可预期的。

- 接口 (interface)，一般谈的是函数的签名 (signature) 或 class 的可访问元素（class 的 “public 接口”)，或是针对某 template 类型参数需为有效的一个表达式，接口完全是指一般性的设计观念。


**命名习惯 Naming Conventions**
- 有意义的名称用于 objects, classes, functions, templates，但某些隐藏于名称背后的意义可能不是那么显而易见，例如两个参数名称
lhs 和 rhs ，分别代表 ”left-hand side“ （左手端）和 “right-hand side“ （右手端），常常以它们作为二元操作符 (binary operators) 函数如 operator=＝ 和 operator* 的参数名称。

- 对于成员函数，左侧实参由 this 指针表现出来，所以有时我单独使用参数名称 rhs 。

- 常将”指向一个 T 型对象”的指针命名为 pt, 意思是 “pointer to T“。

- 对千 references 使用类似习惯： rw 可能是个 reference to wdget, ra 则是个 reference to Airplane 。

- 当讨论成员函数时，偶尔会以 mf 为名。

**关于线程 Threading Consideration**
- 作为一个语言， C++ 对线程 (threads) 没有任何意念，事实上它对任何并发(concurrency) 事物都没有意念。 

- C++标准程序库也一样。当 C+＋受到全世界关注时多线程 (multithreaded) 程序还不存在。

- 线程安全性 (thread safety) 是许多程序员面对的主题。对“标准 C++和真实世界之间的这个缺口”的处理方式是，如果我所检验的 C+＋构件在多线程环境中有可能引发问题，就把它指出来。限制其自身处于单线程考虑之下仍承认多线程的存在。

**TR1 and Boost**
- TRI ("Technical Report 1”) 是一份规范，描述加入 C++ 标准程序库的诸多新机能。这些机能以新的 class templates 和 function templates 形式体现，针对的题目有 hash tables, reference-counting smart pointers, regular expressions, 以及更多。所有 TR1 组件都被置千命名空间七rl 内，后者嵌套千命名空间 std 内。

- Boost 是个组织，亦是一个网站，提供可移植、同僚复审、源码开放的 C+＋程序库。大多数 TRl 机能是以 Boost 的工作为基础。在编译器厂商于其 C++程序库中含入 TR1 之前，对那些搜寻 TR1 实现品的开发人员而言，Boost 网站可能是第一个逗留点。 Boost 提供比 TR1 更多的东西，所以无论如何值得了解它。

- [Boost Site](http://boost.org)


## Chapter 1 让自己习惯 C++; Accustoming Yourself to C++

### 条款 01: 视 C+＋为一个语言联邦; View C++ as a federation of languages

### 条款 02: 尽置以 canst, enum, inline 替换＃define; Prefer consts,enums, and inlines to #defines.

### 条款 03: 尽可能使用 const; Use const whenever possible.

### 条款 04: 确定对象被使用前已先被初始化; Make sure that objects are initialized before they're used.


## Chapter 2 构造／析构／赋值运算 Constructors, Destructors, and Assignment Operators

### 条款 05: 了解 C+＋默默编写并调用哪些函数; Know what functions C++silently writes and calls.

### 条款 06: 若不想使用编译器自动生成的函数，就该明确拒绝; Explicitly disallow the use of compiler-generated functions you do not want.

### 条款 07: 为多态基类声明 virtual 析构函数; Declare destructors virtual in polymorphic base classes.

### 条款 08: 别让异常逃离析构函数; Prevent exceptions from leaving destructors.

### 条款 09: 绝不在构造和析构过程中调用 virtual 函数; Never call virtual functions during construction or destruction.

### 条款 10: 令 operator= 返回一个 reference to *this; Have assignment operators return a reference to *this.

### 条款 11 ：在 operator= 中处理“自我赋值”; Handle assignment to self in operator=.

### 条款 12: 复制对象时勿忘其每一个成分; Copy all parts of an object.

## Chapter 3 资源管理 Resource Management

### 条款 13 ：以对象管理资源; Use objects to manage resources.

### 条款 14 ：在资源管理类中小心 copying 行为; Think carefully about copying behavior in resource-managing classes.

### 条款 15 ：在资源管理类中提供对原始资源的访问; Provide access to raw resources in resource-managing classes.

### 条款 16: 成对使用 new 和 delete 时要采取相同形式; Use the same form in corresponding uses of new and delete.

### 条款 17: 以独立语句将 newed 对象宣入智能指针; Store newed objects in smart pointers in standalone statements.


## Chapter 4 话阳与声明 Designs and Declarations

### 条款 18: 让接口容易被正确使用，不易被误用; Make interfaces easy to use correctly and hard to use incorrectly.

### 条款 19: 设计 class 犹如设计 type; Treat class design as type design.

### 条款 20 ：宁以 pass-by-reference-to-canst 替换 pass-by-value; Prefer pass-by-reference-to-canst to pass-by-value.

### 条款 21: 必须返回对象时，别妄想返回其 reference; Don't try to return a reference when you must return an object.

### 条款 22: 将成员变置声明为 private; Declare data members private.

### 条款 23: 宁以 non-member、 non-friend 替换 member 函数; Prefer non-member non-friend functions to member functions.

### 条款 24: 若所有参数皆需类型转换，请为此采用 non-member 函数; Declare non-member functions when type conversions should apply to all parameters.

### 条款 25: 考虑写出一个不抛异常的 swap 函数; Consider support for a non-throwing swap.


## Chapter 5 实现 Implementations

### 条款 26: 尽可能延后变噩定义式的出现时间; Postpone variable definitions as long as possible.

### 条款 27: 尽置少做转型动作; Minimize casting.

### 条款 28: 避免返回 handles 指向对象内部成分; Avoid returning "handles" to object internals.

### 条款 29: 为“异常安全”而努力是值得的; Strive for exception-safe code.

### 条款 30: 透彻了解 inlining 的里里外外; Understand the ins and outs of inlining.

### 条款 31 ：将文件间的编译依存关系降至最低; Minimize compilation dependencies between files.


## Chapter 6 继承与面向对象设计 Inheritance and Object-Oriented Design

### 条款 32: 确定你的 public 继承塑模出 is-a 关系; Make sure public inheritance models "is-a."

### 条款 33: 避免遮掩继承而来的名称; Avoid hiding inherited names.

### 条款 34: 区分接口继承和实现继承; Differentiate between inheritance of interface and inheritance of implementation.

### 条款 35: 考虑 virtual 函数以外的其他选择; Consider alternatives to virtual functions.

### 条款 36: 绝不重新定义继承而来的 non-virtual 函数; Never redefine an inherited non-virtual function

### 条款 37: 绝不重新定义继承而来的缺省参数值; Never redefine a function's inherited default parameter value.

### 条款 38: 通过复合塑模出 has-a 或“根据某物实现出'; Model "has-a" or "is-implemented-in-terms-of'through composition.

### 条款 39: 明智而审慎地使用 private 继承; Use private inheritance judiciously

### 条款 40: 明智而审慎地使用多重继承; Use multiple inheritance judiciously


## Chapter 7 梧财反与泛型编程 Templates and Generic Programming

### 条款 41: 了解隐式接口和编译期多态; Understand implicit interfaces and compile-time polymorphism.

### 条款 42: 了解七 typename 的双重意义; Understand the two meanings of typename.

### 条款 43: 学习处理模板化基类内的名称; Know how to access names in templatized base classes

### 条款 44: 将与参数无关的代码抽离 templates; Factor parameter-independent code out of templates.

### 条款 45: 运用成员函数模板接受所有兼容类型; Use member function templates to accept "all compatible types."

### 条款 46: 需要类型转换时请为模板定义非成员函数; Define non-member functions inside templates when type conversions are desired

### 条款 47: 请使用 traits classes 表现类型信患; Use traits classes for information about types.

### 条款 48: 认识 template 元编程; Be aware of template metaprogramming.


## Chapter 8 定制 new 和 delete Customizing new and delete

### 条款 49: 了解 new-handler 的行为; Understand the behavior of the new-handler.

### 条款 50: 了解 new 和 delete 的合理替换时机; Understand when it makes sense to replace new and delete

### 条款 51: 编写 new 和 delete 时需固守常规; Adhere to convention when writing new and delete.

### 条款 52 ：写了 placement new 也要写 placement delete; Write placement delete if you write placement new.


## Chapter 9 杂项讨论 Miscellany

### 条款 53 ：不要轻忽编译器的警告; Pay attention to compiler warnings.

### 条款 54: 让自己熟悉包括 TRl 在内的标准程序库; Familiarize yourself with the standard library, including TRI.

### 条款 55: 让自己熟悉 Boost; Familiarize yourself with Boost.
