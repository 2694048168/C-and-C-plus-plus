/**
 * @file 04_selectionStatements.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <stdexcept>
#include <type_traits>

/**
 * @brief Selection Statements
 * 选择语句表示条件控制流,有两种类型: if 语句和 switch 语句.
 * 
 * ?初始化语句和 if/Initialization Statements and if
 * 在 if 和 else if 声明中添加一个初始化语句 init-statement,
 * 将对象的作用域绑定到 if 语句, 此模式与结构化绑定一起使用, 可以优雅地处理错误.
 * 
 * ?constexpr if 语句/constexpr if Statements
 * 将 if 语句设置为 constexpr, 这样的语句称为 constexpr if 语句.
 * constexpr if 语句在编译时求值, 与 true 条件相对应的代码块将执行, 其余代码块则被忽略.
 * *与模板和＜type_traits＞头文件结合使用时, constexpr if 语句功能是非常强大的.
 * constexpr if 的主要用途是根据类型参数的某些属性在函数模板中提供自定义行为.
 * 
 * =====与 if 语句加初始化列表语法一样,
 * *在初始化表达式中初始化的任何对象都将绑定到switch 语句的作用域.
 * 
 */
template<typename T>
constexpr const char *sign(const T &x)
{
    const char *result{};
    if (x == 0)
    {
        result = "zero";
    }
    else if (x > 0)
    {
        result = "positive";
    }
    else
    {
        result = "negative";
    }
    return result;
}

struct TextFile
{
    bool        success;  // 告知调用者函数调用是否成功
    const char *contents; // 文件内容
    size_t      n_bytes;  // 文件大小
};

TextFile read_text_file(const char *path)
{
    static const char contents[]{"Sometimes the goat is you."};
    return TextFile{true, contents, sizeof(contents)};
}

/**
 * @brief 函数模板 value_of 接受指针, 引用和值,
 * 根据参数是哪种对象, value_of 返回指向的值或值本身.
 * 
 */
template<typename T>
auto value_of(T x)
{
    if constexpr (std::is_pointer<T>::value)
    {
        if (!x)
            throw std::runtime_error{"Null pointer dereference."};
        return *x;
    }
    else
    {
        return x;
    }
}

enum class Color : int
{
    Mauve = 0,
    Pink,
    Russet,

    NUM_COLOR
};

struct Result
{
    const char *name;
    Color       color;
};

Result observe_shrub(const char *name)
{
    return Result{name, Color::Russet};
}

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("float 100 is %s\n", sign(100.0f));
    printf("int -200 is %s\n", sign(-200));
    printf("char 0 is %s\n", sign(char{}));

    // 结构化绑定声明移至 if 语句的初始化语句部分,
    // 这将每个解压缩的对象（success、txt 和 len）的作用域限定在 if 块
    if (const auto [success, txt, len] = read_text_file("README.txt"); success)
    {
        printf("Read %llu bytes: %s\n", len, txt);
    }
    else
    {
        printf("Failed to open README.txt.\n");
    }

    /**
 * @brief 就会发现在编译时求值可以通过消除魔数而大大简化程序;
 * 在其他例子中, 编译时求值也很普遍, 尤其是在创建供他人使用的库时,
 * 因为库编写者通常不知道用户使用库的所有方式, 所以需要编写泛型代码,
 * 通常会使用第模板技术以实现编译时多态, 诸如 constexpr 之类的结构在编写此类代码时会有所帮助.
 * 
 * TODO:如果具有 C 语言的背景知识, 将立刻意识到编译时求值的功能, 它几乎完全取代了对预处理器宏的需求.
 * 
 */
    unsigned long level{8998};
    auto          level_ptr = &level;
    auto         &level_ref = level;
    printf("Power level = %lu\n", value_of(level_ptr));
    ++*level_ptr;
    printf("Power level = %lu\n", value_of(level_ref));
    ++level_ref;
    printf("It's over %lu!\n", value_of(level++));
    try
    {
        level_ptr = nullptr;
        value_of(level_ptr);
    }
    catch (const std::exception &e)
    {
        printf("Exception: %s\n", e.what());
    }

    // venerable switch statement,
    // delves into the addition of the initialization statement into switch
    const char *description;
    switch (const auto result = observe_shrub("Zaphod"); result.color)
    {
    case Color::Mauve:
    {
        description = "mauvey shade of pinky russet";
        break;
    }
    case Color::Pink:
    {
        description = "pinky shade of mauvey russet";
        break;
    }
    case Color::Russet:
    {
        description = "russety shade of pinky mauve";
        break;
    }
    default:
    {
        description = "enigmatic shade of whitish black";
    }
    }

    printf(
        "The other Shaltanac's joopleberry shrub is "
        "always a more %s.",
        description);

    return 0;
}
