/**
 * @file 22_invokeFunctions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>

void greet(const std::string &name)
{
    std::cout << "Welcome, " << name << "\n";
}

class Pet
{
public:
    Pet(std::string name)
        : name_(name)
    {
    }

    void makeSound(const std::string &sound) const
    {
        std::cout << name_ << " happy to said: " << sound << " \n";
    }

    void performTrick(const std::string &trick) const
    {
        std::cout << name_ << " playing on: " << trick << " \n";
    }

    void setMood(const std::string &mood)
    {
        std::cout << name_ << " feel now " << mood << "\n";
    }

    void eat(const std::string &food, int amount)
    {
        std::cout << name_ << " eared " << amount << " count " << food << " \n";
    }

private:
    std::string name_;
};

// 函数对象(Function Objects)的使用 🎯
struct Multiplier
{
    int operator()(int x, int y) const
    {
        return x * y;
    }
};

// 访问类的成员变量 🏗️
class Student
{
public:
    Student(std::string name, int score)
        : name_(name)
        , score_(score)
    {
    }

    std::string name_;
    int         score_;
};

// 在算法中的应用 🔄
class Person
{
public:
    Person(std::string name, int age)
        : name_(name)
        , age_(age)
    {
    }

    std::string getName() const
    {
        return name_;
    }

private:
    std::string name_;
    int         age_;
};

// 高级用法：完美转发 🚀
template<typename Callable, typename... Args>
auto wrapper(Callable &&func, Args &&...args)
{
    return std::invoke(std::forward<Callable>(func), std::forward<Args>(args)...);
}

/* 注意事项和最佳实践 ⚠️
* std::invoke 在处理成员函数指针时需要注意对象的生命周期
* 推荐在泛型编程中使用 std::invoke 来统一处理各种可调用对象
* 配合 std::invoke_result 可以在编译期获取调用结果的类型 */
template<typename F, typename... Args>
auto safe_call(F &&f, Args &&...args)
{
    using result_type = std::invoke_result_t<F, Args...>;
    try
    {
        return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    }
    catch (const std::exception &e)
    {
        std::cerr << "call failed: " << e.what() << " \n";
        return result_type{};
    }
}

// ?std::invoke 和 std::apply 的区别与使用场景
// 通用的函数包装器示例
template<typename F, typename... Args>
auto invoke_wrapper(F &&f, Args &&...args)
{
    return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
}

template<typename F, typename Tuple>
auto apply_wrapper(F &&f, Tuple &&t)
{
    return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}

// ------------------------------------------------
int main(int /* argc */, const char * /* argv[] */)
{
    std::cout << "---------------------\n";
    // 使用 std::invoke 执行函数
    std::invoke(greet, "Wei Li");
    // 直接调用和使用 std::invoke 是等价的
    greet("Wei Li");

    /* std::invoke 的强大之处在于它能统一处理各种不同类型的函数调用方式 🚀
    ✅ 普通函数
    ✅ 成员函数
    ✅ Lambda表达式
    ✅ 函数对象 */
    Pet cat("Mimi");

    // 使用 std::invoke 让猫咪叫声 🔊
    std::invoke(&Pet::makeSound, cat, "mi-mimi~");
    // 让猫咪表演节目 🎪
    std::invoke(&Pet::performTrick, cat, "fan-gen-tou");

    Pet dog("WanWan");
    // 使用 std::invoke 调用各种成员函数 🎮
    std::invoke(&Pet::setMood, dog, "very happy");
    std::invoke(&Pet::eat, dog, "dog-food", 2);

    // Lambda 表达式和 std::invoke 的完美配合
    std::cout << "---------------------\n";
    // 创建一个可爱的计算器 lambda 🧮
    auto calculator = [](int a, int b)
    {
        return a + b;
    };

    // 使用 std::invoke 来调用我们的计算器 🎯
    int result = std::invoke(calculator, 40, 2);
    std::cout << "The result of Computer: " << result << " \n";

    int  multiplier = 10; // 外部变量 🔢
    // 创建一个捕获外部变量的 lambda ⚡
    auto multiply_by = [multiplier](int x)
    {
        return x * multiplier;
    };

    // 使用 std::invoke 调用带状态的 lambda 🎯
    result = std::invoke(multiply_by, 5);
    std::cout << "5 x 10 = " << result << " \n";

    /* 使用 lambda 配合 std::invoke 时,代码更加灵活清晰 🎨
    1. 可以轻松处理各种类型的参数和返回值 🎁
    2. 特别适合临时性的函数操作 ⚡
    3. 让代码更具可读性和维护性 📚
    记住,std::invoke 就像是一个魔法棒 🪄,可以优雅地调用任何 lambda 表达式!
    
    实践建议 🌟
    1. 保持 lambda 表达式简短清晰 📝
    2. 适当使用注释说明 lambda 的功能 💭
    3. 合理使用参数和返回值类型 🎯
    4. 注意捕获列表的使用  */
    std::vector<int> numbers{1, 2, 3, 4, 5};

    // 创建一个变换数字的 lambda 🔄
    auto double_it = [](int x)
    {
        return x * 2;
    };

    // 使用 std::invoke 配合 transform 算法 ✨
    std::transform(numbers.begin(), numbers.end(), numbers.begin(),
                   [&double_it](int x)
                   {
                       return std::invoke(double_it, x); // 调用 lambda 🎯
                   });

    for (int n : numbers)
    {
        std::cout << n << " ";
    }
    std::cout << "  \n";

    std::cout << "---------------------\n";
    Multiplier mult;
    // 使用 std::invoke 调用函数对象
    result = std::invoke(mult, 6, 7);
    std::cout << "6 * 7 = " << result << " \n";

    std::cout << "---------------------\n";
    Student     student("Hong Xia", 95);
    // 使用 std::invoke 访问成员变量
    std::string name  = std::invoke(&Student::name_, student);
    int         score = std::invoke(&Student::score_, student);
    std::cout << name << " score is : " << score << " \n";

    std::cout << "---------------------\n";
    std::vector<Person> people{Person("Zha", 25), Person("Li", 30), Person("Wang", 28)};

    // 使用 std::invoke 配合算法
    std::vector<std::string> names;
    std::transform(people.begin(), people.end(), std::back_inserter(names),
                   [](const Person &p) { return std::invoke(&Person::getName, p); });
    for (const auto &elem : names)
    {
        std::cout << elem << " ";
    }

    std::cout << "\n---------------------\n";
    auto lambda = [](int x, int y)
    {
        return x + y;
    };
    result = wrapper(lambda, 10, 20);
    std::cout << "10 + 20 = " << result << " \n";

    return 0;
}
