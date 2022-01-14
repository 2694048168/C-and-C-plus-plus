/**
 * @file rvalue_reference.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief rvalue reference and move semantic and perfect forwarding
 * @version 0.1
 * @date 2022-01-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <vector>
#include <string>

/** lvalue, rvalue, prvalue, xvalue
 * To understand what the rvalue reference is all about, clear understanding of the lvalue and rvalue.
 * 1. lvalue, left value, as the name implies, is the value to the left of the assignment symbol. 
 *      To be precise, an lvalue is a persistent object that still exists after an expression
 *       (not necessarily an assignment expression).
 * 2. rvalue, right value, the value on the right refers to the temporary object 
 *      that no longer exists after the expression ends.
 * 
 * In C++11,  in order to introduce powerful rvalue references, the concept of rvalue values is further
 *  divided into: prvalue, and xvalue.
 * 3. prvalue, pure right value, purely right value, either purely literal, such as "10", "true";
 *      either the result of the evaluation is equivalent to a literal or anonymous temporary object, 
 *      for example "1+2". Temporary variables returned by non-references, temporary variables generated
 *      by operation expression, original literals, and Lambda expression are all pure right values.
 * 4. xvalue, expiring value is the concept proposed by C++11 to introduce rvalue reference
 *      (so in traditional C++, pure rvalue and rvalue are the same concepts),
 *      a value that is destroyed but can be moved.
 * 
 */
// pure right value
class Function_prvalue
{
private:
    const char *&&right = "this is a rvalue";

public:
    void Function_bar()
    {
        right = "still rvalue"; /* the string literal is a rvalue */
    }
};

// expiring value (rvalue reference)
std::vector<int> function_xvalue()
{
    std::vector<int> temp = {1, 2, 3, 4};
    return temp;
}

/* rvalue reference and lvalue reference
    To get a xvalue, need to use the declaration of the rvalue reference: T &&, where T is the type.
    The statement of the rvalue reference extends the lifecycle of this temporary value, 
    and as long as the variable is alive, the xvalue will continue to survive.

    C++11 provides the std::move method to unconditionally convert lvalue parameters to rvalues.
    With it we can easily get a rvalue temporary object, for example:
*/
void reference(std::string &str)
{
    std::cout << "lvalue" << std::endl;
}

void reference(std::string &&str)
{
    std::cout << "rvalue" << std::endl;
}

int main(int argc, char **argv)
{
    // pure right value
    const char *const &left = "this is an lvalue"; /* the string literal is an lvalue */

    // expiring value (destoryed immediately and being able to be moved)
    std::vector<int> vec = function_xvalue();
    for (auto element_itr = vec.begin(); element_itr != vec.end(); ++element_itr)
    {
        std::cout << *element_itr << ' ';
    }
    std::cout << std::endl;

    // rvalue reference
    std::string lvalue_1 = "string,";             /* lvalue_1 is a left value */
    // std::string &&rvalue_1 = lvalue_1;            /* illegal, rvalue can not refer to lvalue */
    std::string &&rvalue_1 = std::move(lvalue_1); /* legal, std::move can convert lvalue to rvalue */
    std::cout << rvalue_1 << std::endl;

    const std::string &lvalue_2 = lvalue_1 + lvalue_1; /* legal, const lvalue reference can extend temp variable's */
    // lvalue_2 += "Test";                                /* illegal, const reference can not be modified */
    std::cout << lvalue_2 << std::endl;

    std::string &&rvalue_2 = lvalue_1 + lvalue_2; /* legal, rvalue reference extend lifecycle */
    rvalue_2 += "string";                         /* legal, non-const reference can be modified */
    std::cout << rvalue_2 << std::endl;

    reference(rvalue_2); /* output: lvalue */

    return 0;
}
