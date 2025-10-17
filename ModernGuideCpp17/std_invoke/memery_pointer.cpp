/**
 * @file memery_pointer.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-10-17
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <functional>
#include <iostream>
#include <memory>
#include <string>

/**
 * @brief 调用成员函数和访问成员变量
 * std::invoke 优雅地处理了与成员指针相关的调用,
 * 无论提供的是对象实例、对象引用、智能指针还是裸指针.
 * 
 * g++ memery_pointer.cpp -std=c++20
 * clang++ memery_pointer.cpp -std=c++20
 */

struct Widget
{
    std::string id = "default-widget";

    /**
     * @brief 设置 Widget 的 ID
     * @param new_id 新的 ID
     */
    void set_id(const std::string &new_id)
    {
        id = new_id;
        std::cout << "ID set to: " << id << std::endl;
    }

    /**
     * @brief 打印当前 ID
     */
    void print_id() const
    {
        std::cout << "Current ID: " << id << std::endl;
    }
};

// -------------------------------------
int main(int argc, const char *argv[])
{
    Widget  w;
    Widget &w_ref       = w;
    Widget *w_ptr       = &w;
    auto    w_smart_ptr = std::make_unique<Widget>();

    // --- 调用成员函数 ---
    std::cout << "--- Calling Member Functions ---\n";

    // 1. 通过对象实例调用
    std::invoke(&Widget::set_id, w, "id_by_value");

    // 2. 通过对象引用调用
    std::invoke(&Widget::set_id, w_ref, "id_by_ref");

    // 3. 通过对象指针调用
    std::invoke(&Widget::set_id, w_ptr, "id_by_ptr");

    // 4. 通过智能指针调用 (C++17 对 std::invoke 的增强)
    std::invoke(&Widget::set_id, w_smart_ptr, "id_by_smart_ptr");

    // --- 访问成员变量 ---
    std::cout << "\n--- Accessing Member Variables ---\n";

    // 1. 通过对象实例访问
    std::string &id_ref1 = std::invoke(&Widget::id, w);
    std::cout << "Accessed via value: " << id_ref1 << std::endl;
    id_ref1 = "modified_by_value"; // 修改会影响原始对象
    w.print_id();

    // 2. 通过对象指针访问
    std::string &id_ref2 = std::invoke(&Widget::id, w_ptr);
    std::cout << "Accessed via ptr: " << id_ref2 << std::endl;

    // 3. 通过智能指针访问
    std::string &id_ref3 = std::invoke(&Widget::id, w_smart_ptr);
    std::cout << "Accessed via smart_ptr: " << id_ref3 << std::endl;

    return 0;
}
