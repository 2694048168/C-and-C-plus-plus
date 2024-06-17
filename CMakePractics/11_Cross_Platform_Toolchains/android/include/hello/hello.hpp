/**
 * @file hello.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#pragma once

#include <string>

namespace hello {
/// Example class that is explicitly exported into a dll
class Hello
{
public:
    Hello(const std::string &name)
        : name_{name}
    {
    }

    void greet() const;

private:
    const std::string name_;
};
} // namespace hello