## Chapter 2 构造／析构／赋值运算 Constructors, Destructors, and Assignment Operators

### 条款 05: 了解 C+＋默默编写并调用哪些函数; Know what functions C++silently writes and calls.

- 什么时候 empty class （空类）不再是个 empty class 呢？当 C+＋处理过它之后。是的，如果你自己没声明，编译器就会为它声明（编译器版本的）一个 copy构造函数、一个 copy assignment操作符和一个析构函数。此外如果你没有声明任何构造函数，编译器也会为你声明一个 default 构造函数。所有这些函数都是 public 且 inline（见条款 30) .

- 注意，编译器产出的析构函数是个 non-virtual （见条款 7) ，除非这个 class 的 base class 自身声明有 virtual 析构函数（这种情况下这个函数的虚属性； virtualness; 主要来自 base class) 。

- 请记住, 编译器可以暗自为 class 创建 default 构造函数、 copy构造函数、 copy assignment 操作符，以及析构函数。


### 条款 06: 若不想使用编译器自动生成的函数，就该明确拒绝; Explicitly disallow the use of compiler-generated functions you do not want.

- 答案的关键是，所有编译器产出的函数都是 public 。为阻止这些函数被创建出来，你得自行声明它们，但这里并没有什么需求使你必须将它们声明为 public 。因此你可以将 copy构造函数或 copy assignment 操作符声明为 private。藉由明确声明一个成员函数，你阻止了编译器暗自创建其专属版本；而令这些函数为 private, 使你得以成功阻止人们调用它。

- 请记住, 为驳回编译器自动（暗自）提供的机能，可将相应的成员函数声明为 private 并且不予实现。


### 条款 07: 为多态基类声明 virtual 析构函数; Declare destructors virtual in polymorphic base classes.

- 可以设计 factory （工厂）函数，返回指针指向一个计时对象。 Factory 函数会“返回一个 base class 指针，指向新生成之 derived class 对象”.

- 给 base class 一个 virtual 析构函数。此后删除 derived class 对象就会如你想要的那般.

- 欲实现出 virtual 函数，对象必须携带某些信息，主要用来在运行期决定哪一个 virtual 函数该被调用。这份信息通常是由一个所谓 vptr (virtual table pointer) 指针指出。 vptr 指向一个由函数指针构成的数组，称为 vtbl (virtual table) ；每一个带有 virtual 函数的 class 都有一个相应的 vtbl 。当对象调用某一 virtual 函数，实际被调用的函数取决于该对象的 vptr 所指的那个 vtbl一编译器在其中寻找适当的函数指针。

- 有时候令 class 带一个 pure virtual 析构函数.  pure virtual 函数导致 abstract （抽象） classes 也就是不能被实体化 (instantiated) 的 class 。

- polymorphic （带多态性质的） base classes 应该声明一个 virtual 析构函数。如果 class 带有任何 virtual 函数，它就应该拥有一个 virtual 析构函数。

- Classes 的设计目的如果不是作为 base classes 使用，或不是为了具备多态性 (polymorphically) ，就不该声明 virtual 析构函数。


### 条款 08: 别让异常逃离析构函数; Prevent exceptions from leaving destructors.

- 如果程序遭遇一个“于析构期间发生的错误”后无法继续执行， “强迫结束程序”是个合理选项。毕竟它可以阻止异常从析构函数传播出去（那会导致不明确的行为）。也就是说调用 std::abort 可以抢先制“不明确行为“于死地。

- 析构函数绝对不要吐出异常。如果一个被析构函数调用的函数可能抛出异常，析构函数应该捕捉任何异常，然后吞下它们（不传播）或结束程序。

- 如果客户需要对某个操作函数运行期间抛出的异常做出反应，那么 class 应该提供一个普通函数（而非在析构函数中）执行该操作。


### 条款 09: 绝不在构造和析构过程中调用 virtual 函数; Never call virtual functions during construction or destruction.

- 在构造和析构期间不要调用 virtual 函数，因为这类调用从不下降至 derived class （比起当前执行构造函数和析构函数的那层）.


### 条款 10: 令 operator= 返回一个 reference to *this; Have assignment operators return a reference to *this.

- 为了实现“连锁赋值”，赋值操作符必须返回一个 reference 指向操作符的左侧实参。

- 请记住，令赋值 (assignment) 操作符返回一个 reference to *this.


### 条款 11 ：在 operator= 中处理“自我赋值”; Handle assignment to self in operator=.

- 自我赋值。这些并不明显的自我赋值，是“别名” (aliasing) 带来的结果：所谓“别名”就是“有一个以上的方法指称（指涉）某对象”。

- 确保当对象自我赋值时 operator= 有良好行为。其中技术包括比较”来源对象”和“目标对象”的地址、精心周到的语句顺序、以及 copy-and-swap 。

- 确定任何函数如果操作一个以上的对象，而其中多个对象是同一个对象时，其行为仍然正确。


### 条款 12: 复制对象时勿忘其每一个成分; Copy all parts of an object.

-  copy assignment 操作符
- copy 构造函数
- Copying 函数应该确保复制”对象内的所有成员变量”及“所有 base class 成分”。
- 不要尝试以某个 copying 函数实现另一个 copying 函数。应该将共同机能放进第三个函数中，并由两个 coping 函数共同调用。