/**
 * @file 14_StaticPolymorphism.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <string>

/**
* @brief CRTP（Curiously Recurring Template Pattern）是一种 C++ 编程技巧
* 使用模板类和继承的组合来实现静态多态, 该模式的关键思想是: 在模板类的定义中,
* 模板参数是当前类自身（通常是派生类）; 这个技巧通常用于实现编译时多态，优化性能
* 
* 使用了 static_cast 进行类型转换，根据 CRTP 的定义，
* 在 Info 的派生类中调用 getClassName 接口，并且 T 就是这里的派生类，
* 这里的 static_cast 转换一定是合法的，因为这里的 this 就是派生类型 T
* 
*/
template<typename T>
class Info
{
public:
    [[nodiscard]] std::string getClassName()
    {
        return static_cast<T *>(this)->getClassNameImpl();
    }
};

class C : public Info<C>
{
public:
    [[nodiscard]] std::string getClassNameImpl()
    {
        return "C";
    }
};

class D : public Info<D>
{
public:
    [[nodiscard]] std::string getClassNameImpl()
    {
        return "D";
    }
};
