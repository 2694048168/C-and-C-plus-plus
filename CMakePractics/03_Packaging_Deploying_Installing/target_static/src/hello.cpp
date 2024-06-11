#include <hello.hpp>

#include <iostream>

namespace ProjectName { namespace ModuleName {

void greeter::greet()
{
    std::cout << "Hello, world! Static library\n";
}

}} // namespace ProjectName::ModuleName