#pragma once

#include "hello/export_hello.hpp"

#include <string>

namespace hello {
/// Example class that is explicitly exported into a dll
class HELLO_SHARED_EXPORT Hello
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
