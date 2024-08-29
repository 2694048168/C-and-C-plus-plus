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
