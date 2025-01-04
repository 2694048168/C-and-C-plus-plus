/**
 * @file 23_byteBinary.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstddef> /* std::byte */
// enum class byte : unsigned char {};
#include <bitset>
#include <iostream>
#include <utility>
#include <vector>

/* std::byte 与 unsigned char 的关键区别
  本质区别：
    unsigned char: 被视为数值类型，允许算术运算 🔢
    std::byte: 纯粹的字节容器，只支持位运算 🎯
这种限制让代码更安全、语义更清晰！ */

// 实战小案例: 玩转权限控制
// 权限小精灵们 🧚‍♂️
enum class Permissions : unsigned char
{
    None    = 0x0000, // 啥也不能干 🚫
    Read    = 0x0001, // 可以偷看 👀
    Write   = 0x0002, // 可以写字 ✍️
    Execute = 0x0004  // 可以跑起来 🏃‍♂️
};

// ------------------------------------------------
int main(int /* argc */, const char * /* argv[] */)
{
    std::cout << "---------------------\n";
    // unsigned char - 可以进行算术运算 🚫
    unsigned char old = 42;
    old               = old + 1; // 允许，但这对字节操作来说不合理！

    // std::byte - 只允许位运算 ✨
    std::byte modern{42};
    // modern = modern + 1;  // 编译错误！
    modern = modern | std::byte{1}; // 正确的位运算方式 ✅

    std::cout << "---------------------\n";
    std::byte secret{0b101010}; // 二进制data

    // 位运算大法 🔮
    std::byte mask{0b111000}; // 这是我们的魔法面具 mask
    auto      result = secret & mask;
    std::cout << "The result: " << std::bitset<8>(std::to_integer<int>(result)) << " \n";

    std::cout << "---------------------\n";
    std::byte magic_byte{0x2A};
    std::cout << "The Magic number: " << std::to_integer<int>(magic_byte) << " \n";

    std::cout << "---------------------\n";
    std::byte b{0b00001111};
    std::cout << "The original-binary number: " << std::to_integer<int>(b) << " \n";
    b <<= 1; // 嗖！数字们向左跑 🏃‍♂️ // 左移仙术 ⬅️
    std::cout << "The binary-<< number: " << std::to_integer<int>(b) << " \n";
    b >>= 2; // 唰！数字们向右溜 🏃‍♀️  // 右移神通 ➡️
    std::cout << "The binary->> number: " << std::to_integer<int>(b) << " \n";
    // 三大神器 🔮
    b |= mask; // 或运算：两个数合体 🤝
    std::cout << "The binary-| number: " << std::to_integer<int>(b) << " \n";
    b &= mask; // 与运算：双剑合璧 ⚔️
    std::cout << "The binary-& number: " << std::to_integer<int>(b) << " \n";
    b ^= mask; // 异或运算：完美变身 🦸‍♂️
    std::cout << "The binary-^ number: " << std::to_integer<int>(b) << " \n";

    std::cout << "---------------------\n";
    // 创建一个空权限盒子 📦
    std::byte permissions{0};

    // 往盒子里放入权限 🎁
    permissions |= std::byte{static_cast<unsigned char>(Permissions::Read)};  // 放入读权限
    permissions |= std::byte{static_cast<unsigned char>(Permissions::Write)}; // 放入写权限

    // 偷偷看看有没有读权限 🔍
    bool canRead = (permissions & std::byte{static_cast<unsigned char>(Permissions::Read)}) != std::byte{0};
    std::cout << "can see? " << (canRead ? "OK" : "NO") << "\n";
    /* 权限就像积木块 🧱
     * 用 |= 把权限放进盒子 📥
     * 用 & 来检查权限是否存在 🔍
    一个字节八个位，就能存八种权限，超级省空间！ 🚀
    记住，std::byte 就像一个专业的杂技演员 - 它只做位运算这一件事，但是做得非常专业！
    这就是它的美，简单而纯粹 ✨ */

    std::cout << "---------------------\n";
    // 字节数组操作 - 玩转二进制数据
    std::vector<std::byte> buffer(4); // 4个格子的魔法盒子
    buffer[0] = std::byte{0xFF};      // 第一格放个满值 💎
    buffer[1] = std::byte{0x00};      // 第二格放个空值 🕳️
    for (const auto &b : buffer)
    {
        std::cout << std::to_integer<int>(b) << " ";
    }

    std::cout << "\n---------------------\n";
    // 与其他类型的转换 - 变形记
    int       number = 12345;                  // 原始数字 🔢
    std::byte bytes[sizeof(int)];              // 准备魔法容器 🎁
    std::memcpy(&bytes, &number, sizeof(int)); // 变身开始！ ✨

    // 变身回来 🎭
    int restored;                                // 准备还原容器 📦
    std::memcpy(&restored, &bytes, sizeof(int)); // 还原魔法 🌟

    // 见证奇迹的时刻 🎪
    std::cout << "before: " << number << " \n"
              << "after: " << restored << " \n";

    // 性能考虑 - 快得飞起 🚀
    // 这两行代码就是最好的保证书 📜
    static_assert(sizeof(std::byte) == 1, "std::byte must be one byte");        // 大小刚刚好 📏
    static_assert(alignof(std::byte) == 1, "std::byte memory-align must be 1"); // 对齐完美 ✨

    // 内存小把戏
    std::vector<std::byte> magic(1024);                  // 开启魔法空间 🌟
    std::fill(magic.begin(), magic.end(), std::byte{0}); // 施展清零术 ✨

    return 0;
}
