/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Chapter 1 of Effective C++
 * @version 0.1
 * @date 2022-04-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <iostream>

#define ASPECT_RATIO 1.653
const double AspectRatio = 1.653;

const char *const authorName = "Wei Li";
const std::string author_name("Wei Li");

#define CLAA_WITH_MAX(num_1, num_2) ((num_1) > (num_2) ? (num_1) : (num_2))

template <typename T>
inline auto callWithMax(const T &num_1, const T &num_2)
{
    return num_1 > num_2 ? num_1 : num_2;
}

// const to pointer
char greeting[] = "Hello";
char *ptr_non = greeting;                      /* non-const pointer, non-const data */
const char *ptr_data = greeting;               /* non-const pointer, const data */
char const *ptr_data_ = greeting;              /* non-const pointer, const data */
char *const ptr_pointer = greeting;            /* const pointer, non-const data */
const char *const ptr_data_pointer = greeting; /* const pointer, const data */

int main(int argc, char const *argv[])
{
    printf("The constant with #define is : %f\n", ASPECT_RATIO);
    printf("The constant with const is : %f\n", AspectRatio);

    std::cout << "The author name is : " << authorName << std::endl;
    std::cout << "The author name is : " << author_name << std::endl;

    std::cout << "The maximum number is : " << CLAA_WITH_MAX(4, 42) << std::endl;
    std::cout << "The maximum number is : " << callWithMax(4, 42) << std::endl;

    return 0;
}
