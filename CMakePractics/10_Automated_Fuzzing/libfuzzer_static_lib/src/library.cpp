/**
 * ______________________________________________________
 * Message printer implementation
 *
 * SPDX-License-Identifier:	MIT
 * ______________________________________________________
 */

#include <library/library.hpp>

#include <cstring>


namespace chapter11 { namespace ex01 {

void message_printer::print(const char *msg, std::uint32_t len)
{
    constexpr char a[] = "Hello from CMake Best Practices!";

    if (len < sizeof(a))
    {
        return;
    }

    if (std::memcmp(a, msg, len) == 0)
    {
        volatile char *ptr{nullptr};
        // attempt to write an invalid memory location
        // it is undefined behavior
        *ptr = 'a';
    }
}
}} // namespace chapter11::ex01