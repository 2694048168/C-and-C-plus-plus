## 资源管理 Resource Management

### 条款 13 ：以对象管理资源; Use objects to manage resources.

- 实际上这正是隐身于本条款背后的半边想法：把资源放进对象内，便可倚赖 C++的“析构函数自动调用机制”确保资源被释放。 

- 获得资源后立刻放进管理对象 (managing object) 内。实际上“以对象管理资源”的观念常被称为“资源取得时机便是初始化时机” (Resource
Acquisition Is Initialization; RAIi) ，因为几乎总是在获得一笔资源后于同一语句内以它初始化某个管理对象。有时候获得的资源被拿来赋值（而非初始化）某个管理对象，但不论哪一种做法，每一笔资源都在获得的同时立刻被放进管理对象中。

- 管理对象 (managing object) 运用析构函数确保资源被释放。不论控制流如何离开区块，一旦对象被销毁（例如当对象离开作用域）其析构函数自然会被自动调用，于是资源被释放。如果资源释放动作可能导致抛出异常，事情变得有点棘手，但条款 8 已经能够解决这个问题，所以这里我们也就不多操心了。

- 为防止资源泄漏，请使用 RAII 对象，它们在构造函数中获得资源并在析构函数中释放资源。

- 两个常被使用的 RAII classes 分别是 trl::shared_ptr 和 auto_ptr。前者通常是较佳选择，因为其 copy 行为比较直观。若选择 auto_ptr, 复制动作会使它（被复制物）指向 null 。


### 条款 14 ：在资源管理类中小心 copying 行为; Think carefully about copying behavior in resource-managing classes.

 