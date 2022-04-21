## Chapter 1 让自己习惯 C++ Accustoming Yourself to C++

### 条款 01: 视 C+＋为一个语言联邦; View C++ as a federation of languages

- Exceptions （异常）对函数的结构化带来不同的做法

- templates （模板）将我们带到新的设计思考方式

- STL 则定义了一个前所未见的伸展性做法。

- 多重范型编程语言 (multi-paradigm programming language) ，一个同时支持过程形式 (procedural) 、面向对象形式 (object-oriented) 、函数形式 (functional) 、泛型形式 (generic) 、元编程形式 (meta-programming)的语言。

- 为了理解 C++, 必须认识其主要的次语言。记住这四个次语言，当你从某个次语言切换到另一个，导致高效编程守则要求你改变策略时，不要感到惊讶。
    1. C ：说到底 C++ 仍是以 C 为基础。区块 (blocks) 、语句 (statements) 、预处理器 (preprocessor) 、内置数据类型 (built-in data types) 、数组 (arrays) 、指针 (pointers) 等统统来自 C。许多时候 C++对问题的解法其实不过就是较高级的 C 解法，但当你以 C++内的 C 成分工作时，高效编程守则映照出 C 语言的局限：没有模板 (templates) ，没有异常 (exceptions) ，没有重载 (overloading)
    2. Object-Oriented C++ : 这部分也就是 C with Classes 所诉求的： classes （包括构造函数和析构函数），封装 (encapsulation) 、继承 (inheritance) 、多态(polymorphism) 、 virtual 函数（动态绑定）等等。这一部分是面向对象设计之古典守则在 C++上的最直接实施。
    3. Template C++ : 这是 C++的泛型编程 (generic programming) 部分，Template 相关考虑与设计已经弥漫整个 C++，良好编程守则中“惟 template 适用”的特殊条款并不罕见（例如条款 46 谈到调用 template functions 时如何协助类型转换）。实际上由于templates 威力强大，它们带来崭新的编程范型 (programming p~radigm) ，也就是所谓的 template metaprogramming CTMP, 模板元编程）。条款 48 对此提供了一份概述，但除非你是 template 激进团队的中坚骨干，大可不必太担心这些。 TMP 相关规则很少与 C++ 主流编程互相影响。
    4. STL ： STL 是个 template 程序库，它对容器 (containers) 、迭代器 (iterators) 、算法 (algorithms) 以及函数对象 (function objects) 的规约有极佳的紧密配合与协调，然而 templates 及程序库也可以其他想法建置出来。 STL 有自己特殊的办事方式，当你伙同 STL 一起工作，你必须遵守它的规约。

- C++ 高效编程守则视状况而变化，取决于你使用 C++ 的哪一部分。


### 条款 02: 尽置以 canst, enum, inline 替换＃define; Prefer consts,enums, and inlines to #defines.

- 宁可以编译器替换预处理器，因为或许＃define 不被视为语言的一部分

- #define 声明的记号名称也许从未被编译器看见；也许在编译器开始处理源码，之前它就被预处理器移走了。有可能没进入记号表(symbol table) 内。当你运用此常量但获得一个编译错误信息时，可能会带来困惑。这个问题也可能出现在记号式调试器 (symbolic debugger) 中，原因相同：你所使用的名称可能并未进入记号表 (symbol table) 。

- 浮点常量 (floating point constant)
- 常量指针 (constant pointers)

- 将常量的作用域 (scope) 限制于 class 内，你必须让它成为 class 的一个成员 (member) ；而为确保此常蜇至多只有一份实体，你必须让它成为一个 static 成员。

- 可改用所谓的 "the enum hack" 补偿做法。其理论基础是： “一个属千枚举类型 (enumerated type) 的数值可权充 ints 被使用”
-  实用主义角度，“enum hack" 是 template meta-programming （模板元编程) 的基础技术。

- 获得宏带来的效率以及一般函数的所有可预料行为和类型安全性 (type safety），只要你写出 template inline 函数

- 对于单纯常量，最好以 const 对象或 enums 替换 #defines
- 对于形似函数的宏 (macros) ，最好改用 inline 函数替换 #defines


### 条款 03: 尽可能使用 const; Use const whenever possible.

- const 的一件奇妙事情是，它允许你指定一个语义约束（也就是指定一个“不该被改动”的对象），而编译器会强制实施这项约束。

- 关键字 const 多才多艺。可以用它在 classes 外部修饰 global 或 namespace 作用域中的常量，或修饰文件、函数、或区块作用域 (block scope) 中被声明为 static 的对象。也可以用它修饰 classes 内部的 static 和 non-static 成员变量。面对指针，也可以指出指针自身、指针所指物，或两者都（或都不）是 const.

- const 语法虽然变化多端，但并不莫测高深。如果关键字 const 出现在星号左边，表示被指物是常量；如果出现在星号右边，表示指针自身是常最；如果出现在星号两边，表示被指物和指针两者都是常量。如果被指物是常量，有些程序员会将关键字 const 写在类型之前，有些人会把
它写在类型之后、星号之前。两种写法的意义相同，两种形式都有人用，你应该试着习惯它们。

- STL 迭代器系以指针为根据塑模出来，所以迭代器的作用就像个 T* 指针。声明迭代器为 const 就像声明指针为 const 一样（即声明一个 T* const 指针），表示这个迭代器不得指向不同的东西，但它所指的东西的值是可以改动的。如果希望迭代器所指的东西不可被改动（即希望 STL 模拟一个 const T*指针），需要的是 const—iterator.

- const 最具威力的用法是面对函数声明时的应用。在一个函数声明式内， const 可以和函数返回值、各参数、函数自身（如果是成员函数）产生关联。

- 令函数返回一个常量值，往往可以降低因客户错误而造成的意外，而又不至于放弃安全性和高效性。

- const 成员函数，将 const 实施于成员函数的目的，是为了确认该成员函数可作用于 const 对象身上。这一类成员函数之所以重要，基于两个理由。第一，它们使 class 接口比较容易被理解。这是因为，得知哪个函数可以改动对象内容而哪个函数不行，很是重要。第二，它们使“操作 const 对象”成为可能。这对编写高效代码是个关键，因为如条款 20 所言，改善 C+＋程序效率的一个根本办法是以 pass by reference-to-const 方式传递对象，而此技术可行的前提是，我们有 const 成员函数可用来处理取得（并经修饰而成）的 const 对象。 许多人漠视一件事实：两个成员函数如果只是常量性 (constness) 不同，可以被重载。这实在是一个重要的 C++ 特性。

- 成员函数如果是 const 意味什么？这有两个流行概念： bitwise constness（physical constness) 和 logical constness

- 利用 C++ 的一个与 const 相关的摆动场： mu七able（可变的）, mutable 释放掉 non-static 成员变量的 bitwise constness 约束
- casting away constness 常量性转除
- const 成员函数承诺绝不改变其对象的逻辑状态(logical state), non-con玩成员函数却没有这般承诺

- 将某些东西声明为 const 可帮助编译器侦测出错误用法。 const 可被施加于任何作用域内的对象、函数参数、函数返回类型、成员函数本体。

- 编译器强制实施 bitwise constness ，但你编写程序时应该使用”概念上的常量性” (conceptual constness)

- 当 const 和 non-const 成员函数有着实质等价的实现时，令 non-const 版本调用 const 版本可避免代码重复


### 条款 04: 确定对象被使用前已先被初始化; Make sure that objects are initialized before they're used.

- 读取未初始化的值会导致不明确的行为，在某些语境下(C part of C++) x 保证被初始化（为 0) ，但在其他语境中(non-C parts of C++)却不保证。

- 一些复杂的规则，描述“对象的初始化动作何时一定发生，何时不一定发生”。

- 最佳处理办法就是：永远在使用对象之前先将它初始化。对于无任何成员的内置类型，你必须手工完成此事。至于内置类型以外的任何其他东西，初始化责任落在构造函数 (constructors) 身上。规则很简单：确保每一个构造函数都将对象的每一个成员初始化。重要的是别混淆了赋值 (assignment) 和初始化 (initialization) 

- 构造函数的一个较佳写法是，使用所谓的 member initialization list （成员初值列）替换赋值动作来完成初始化。由于编译器会为用户自定义类型 (user-defined types) 之成员变量自动调用 default 构造函数。

- C++ 有着十分固定的“成员初始化次序”。是的，次序总是相同： base classes 更早于其 derived classes 被初始化，而 class 的成员变量总是以其声明次序被初始化。

- “不同编译单元内定义之 non-local static 对象”的初始化次序。所谓编译单元 (translation unit) 是指产出单一目标文件 (single object file) 的那些源码。基本上它是单一源码文件加上其所含入的头文件 (#include files) 。

-  non-local static 对象被 local static 对象替换了。 Design Patterns 这是 Singleton 模式的一个常见实现手法。

- 为内置型对象进行手工初始化，因为 C扫．不保证初始化它们。

- 构造函数最好使用成员初值列 (member initialization list) ，而不要在构造函数本体内使用赋值操作 (assignment) 。初值列列出的成员变量，其排列次序应该和它们在 class 中的声明次序相同。

- 为免除“跨编译单元之初始化次序”问题，请以 local static 对象替换 non-local static 对象。