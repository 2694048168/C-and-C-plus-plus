#include "internal.hpp"

#include <iostream>

namespace hello { namespace details {
void print_impl(const std::string &name)
{
    std::cout << "Hello " << name << " from a shared library\n";
}
}} // namespace hello::details
