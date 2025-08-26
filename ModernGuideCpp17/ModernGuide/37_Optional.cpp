/**
 * @file 37_Optional.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-26
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

/**
 * @brief std::optional
 * std::optional 不是万金油, 但在很多特定场景下, 它就是最佳选择.
 * 1. 当函数的返回值 "可有可无" 时：这是最经典的场景; 比如查找、解析、查询等操作, 
 * 找不到或解析失败都是一种意料之中的、正常的结果, 而非程序错误; 作为"可选"的函数参数: 
 * 虽然不那么常用, 但可以传递一个"可选"的配置项.
 * 2. 作为 "可能不存在" 的类成员: 比如一个 User 类, 他的"昵称"(nickname) 
 * 可能是 std::optional<std::string>, 因为不是所有用户都会设置昵称; 
 * 它能让你的代码意图更加清晰, 也从根本上消除了因为忘记检查特殊返回值而导致程序崩溃的风险.
 * 
 */

/**
 * @brief Get the env object
 * 
 * @param name 
 * @return std::optional<std::string> 
 */
std::optional<std::string> get_env(const std::string &name)
{
    const char *value = std::getenv(name.c_str());
    if (value)
    {
        // 找到了！把 C 风格字符串包装成 std::string 放进盒子里返回
        // 注意：这里 optional 内部会构造一个 string，但这是合理的开销
        return std::string(value);
    }
    // 没找到？那就返回一个空盒子
    return std::nullopt;
}

// --------------------------------------
int main(int argc, const char *argv[])
{
    // 咱们来试试找一个几乎所有系统都有的环境变量
    auto path_env = get_env("PATH");

    if (path_env)
    {
        // 就这么简单！就像在问：“盒子里有东西吗？”
        // 如果有，path_env 的布尔判断就是 true
        std::cout << "找到了 PATH 变量！\n";
        // 你可以用 * 号或者 -> 像指针一样取出里面的宝贝
        std::cout << "它的长度是：" << path_env->length() << std::endl;
    }
    else
    {
        std::cout << "奇怪，竟然没有找到 PATH 变量！\n";
    }

    // ----------------------------------------
    auto fake_env = get_env("THIS_VAR_DOES_NOT_EXIST_I_SWEAR");

    if (fake_env)
    {
        // 盒子是空的，这里的代码永远不会被执行，绝对安全！
        std::cout << "这是不可能的！" << *fake_env << std::endl;
    }
    else
    {
        // fake_env 的布尔判断是 false
        std::cout << "很好，没有找到那个不存在的变量！程序很稳定！👍\n";
    }

    // ------------------------
    // 尝试获取程序主题，如果环境变量没设置，就默认使用 "light"
    auto        ret   = get_env("MY_APP_THEME");
    std::string theme = ret.value_or("light");
    std::cout << "当前使用的主题是: " << theme << "!\n";

    // --------------------------------------------
    auto definitely_exists_env = get_env("PATH");
    // 在这一刻，你百分百确定它有值（比如你前面已经用 if 检查过了）
    std::cout << "勇士取出的值: " << definitely_exists_env.value() << std::endl;

    try
    {
        auto maybe_empty_env = get_env("THIS_SHOULD_BE_EMPTY");
        // 💥 BOOM! 如果盒子是空的，调用 value() 会立刻抛出一个异常
        std::cout << maybe_empty_env.value() << std::endl;
    }
    catch (const std::bad_optional_access &e)
    {
        std::cout << "哎呀，盒子是空的，程序爆炸了！😱 " << e.what() << std::endl;
    }

    return 0;
}
