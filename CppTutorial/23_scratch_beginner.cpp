/**
 * @file 23_scratch_beginner.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 现代 C++ 小示例
 * @version 0.1
 * @date 2026-09-16
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <iostream>
#include <memory>
#include <ranges>
#include <vector>

struct Widget
{
    void run() const
    {
        std::cout << "Widget::run\n";
    }
};

// =============================
int main(int argc, char **argv)
{
    // 智能指针：自动管理生命周期
    auto w = std::make_unique<Widget>();
    w->run();

    // CTAD + ranges + lambda
    std::vector v{1, 2, 3, 4, 5, 6};

    auto even_squares
        = v | std::views::filter([](int n) { return n % 2 == 0; }) | std::views::transform([](int n) { return n * n; });

    for (int n : even_squares)
    {
        std::cout << n << ' ';
    }
    std::cout << '\n';
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\23_scratch_beginner.cpp -std=c++23
// g++ .\23_scratch_beginner.cpp -std=c++23
