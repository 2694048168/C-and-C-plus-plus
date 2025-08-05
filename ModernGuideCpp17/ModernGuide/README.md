## Modern C++ Guide and Performance and Paradigm for Beginner

> Modern C++ since C++11, and some features and performance for Beginner

### 稳定性和兼容性 Stability and Compatibility
- 原始字符串的字面量 **00_raw_string.cpp**
- 超长整形 long long **01_long_integer.cpp**
- 类成员的快速初始化 **02_init_members.cpp**
- 类的虚函数final和override **03_final_override.cpp**
- 模板优化(模板右尖括号和默认参数) **04_template_opt.cpp**
- 数值类型和字符串之间的转换 **05_string_number.cpp**
- 静态断言 static_assert **06_static_assert.cpp**
- noexcept修饰函数 **07_noexcept_throw.cpp**

### 易学和易用性 Easy-Learn and User-Friendly
- 自动类型推导 auto and decltype **08_auto_decltype.cpp**
- 基于范围的for循环 **09_for_range.cpp**
- 指针空值类型 [nullptr](https://subingwen.cn/linux/file-descriptor/) **10_nullptr_NULL.cpp**
```C++
#ifndef NULL
    #ifdef __cplusplus
        #define NULL 0
    #else
        #define NULL ((void *)0)
    #endif
#endif
```
- Lambda表达式 **11_lambda_expression.cpp**

### 通用性能的提升 Improvement in Performance
- 常量表达式修饰符 constexpr **12_constexpr_modifier.cpp**
- 委托构造函数和继承构造函数 **13_delegate_inheritance_constructor.cpp**
- 右值引用 R-value reference **14_readValue_reference.cpp**
- 转移和完美转发 move&forward **15_move_forward.cpp**
- 列表初始化 **16_initialization.cpp**
- using的使用 **17_using_alias.cpp**
- 可调用对象包装器、绑定器 **18_functional_bind.cpp**
- Plain Old Data类型 **19_POD_type.cpp**
- 默认函数控制 =default 与 =delete **20_default_delete.cpp**
- 扩展的 friend 语法 **21_friend.cpp**
- 强类型枚举 **22_class_enum.cpp**
- 非受限联合体 **23_unrestricted_union.cpp**

### 安全性 safety
- 共享智能指针 **24_shared_pointer.cpp**
- 独占的智能指针 **25_unique_pointer.cpp**
- 弱引用智能指针 **26_weak_pointer.cpp**

### 多线程 multi-thread
- 处理日期和时间的chrono库 **27_chrono_time.cpp**
- Modern C++线程类 thread **28_thread_use.cpp**
- 线程命名空间 this_thread **29_this_thread.cpp**
- 多线程 call_once **30_call_once.cpp**
- 线程同步之互斥锁 mutex **31_thread_synchronization_mutex.cpp**
- 线程同步之条件变量 **32_thread_synchronization_condition_variable.cpp**
- 线程同步之原子变量 atomic **33_thread_synchronization_atomic.cpp**
- 多线程异步操作 **34_thread_asynchronous.cpp**
- 多线程异步线程池 **35_thread_asynchronous_pool.cpp**
- 委托构造 **36_Delegate_Construct.cpp**
