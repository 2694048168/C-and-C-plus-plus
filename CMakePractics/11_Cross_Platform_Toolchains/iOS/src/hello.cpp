/**
 * @file hello.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "internal.hpp"

#include <hello/hello.hpp>

namespace hello {
void Hello::greet() const
{
    details::print_impl(name_);
}
} // namespace hello