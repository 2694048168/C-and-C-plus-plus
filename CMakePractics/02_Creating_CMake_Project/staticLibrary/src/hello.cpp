#include "internal.hpp"

#include <hello/hello.hpp>

namespace hello {
void Hello::greet() const
{
    details::print_impl(name_);
}
} // namespace hello
