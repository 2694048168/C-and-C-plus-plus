/**
 * @file 12_user_type.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "spdlog/fmt/ostr.h" // 需要包含这个头文件来支持 ostream 操作符
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include <string>

// 定义一个自定义类型
struct UserType
{
    int i;

    UserType(int val)
        : i(val)
    {
    }
};

// 为自定义类型定义 fmt 格式化器
template<>
struct fmt::formatter<UserType> : fmt::formatter<std::string>
{
    // 定义如何格式化 UserType
    auto format(UserType type, fmt::format_context &ctx) const -> decltype(ctx.out())
    {
        return format_to(ctx.out(), "[UserType i={}]", type.i);
    }
};

struct FrameInfo
{
    void        *pData   = nullptr;
    unsigned int width   = 0;
    unsigned int height  = 0;
    unsigned int step    = 0;
    unsigned int encoder = 0;
};

template<>
struct fmt::formatter<FrameInfo> : fmt::formatter<std::string>
{
    auto format(FrameInfo type, fmt::format_context &ctx) const -> decltype(ctx.out())
    {
        return format_to(ctx.out(), "[FrameInfo pData={}\twidth={}\theight={}\tstep{}\tencoder={}]", type.pData,
                         type.width, type.height, type.step, type.encoder);
    }
};

// -----------------------------------
int main(int argc, const char **argv)
{
    /* 13. User-defined types用户定义类型
    * 在 spdlog 中, 可以为自定义类型定义格式化规则, 使其能够直接在日志消息中使用.
    * 这是通过 fmt 库的模板特化来实现的, spdlog 基于 fmt 库构建,
    * 因此可以使用相同的机制来格式化自定义类型.
    ?类似重载运算符 << 的作用, 便于直接 C++ stream 直接使用该类型;
    * 
    */
    // 初始化 spdlog
    spdlog::set_level(spdlog::level::info);                         // 设置日志级别
    spdlog::set_default_logger(spdlog::stdout_color_mt("console")); // 使用彩色控制台输出

    spdlog::info("user defined type: {}", UserType(14));
    spdlog::info("user defined type: {}", UserType(24));
    spdlog::info("user defined type: {}", UserType(42));

    spdlog::info("user defined type: {}", FrameInfo());
    spdlog::info("user defined type: {}", FrameInfo{nullptr, 8192, 2000, 8192, 4500});

    /* 代码解释:
    * 1. 定义自定义类型 my_type:
    * ----该类型包含一个整数成员 i, 并通过构造函数进行初始化;
    * 2. 定义 fmt::formatter 模板特化:
    * ----使用 fmt::formatter<my_type> 模板特化来定义 my_type 的格式化规则,
    * ----format 方法用于定义如何将 my_type 格式化为字符串;
    * ----在 format 方法中, 使用 format_to 将格式化后的字符串写入 ctx.out(),
    * ----这里的格式是 [my_type i={}], 其中 {} 是 my_type 的成员 i 的值.
    * 3. 使用自定义类型的示例函数: 在 user_defined_example 函数中, 
    * ----使用 spdlog::info 记录一个包含 my_type 的日志消息,
    * ----由于已经为 my_type 定义了格式化器, spdlog 可以正确地格式化并输出 my_type 类型的值.
    * 
    * 总结: 通过为自定义类型定义 fmt::formatter, 可以让 spdlog 直接处理这些类型,
    *    并在日志中以自定义的格式输出. 这种机制非常灵活, 适用于各种复杂类型的日志记录需求,
    *    使您的日志记录系统更加通用和强大.
    * 
    */

    return 0;
}
