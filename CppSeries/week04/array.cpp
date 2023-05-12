/**
 * @file array.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief Arrays in C++.
 * @attention Must be careful of out of range
 *
 */

#include <iostream>
#include <cstring>

// float array_sum(const float values[4], size_t length)
// float array_sum(const float values[], size_t length)
float array_sum(const float *values, size_t length)
{
    float sum = 0.0f;
    for (size_t i = 0; i < length; i++)
    {
        sum += values[i];
    }

    return sum;
}

// You must tell the function the bound of an array,
// otherwise, elements cannot be accessed
// if the array is a variable-length one, it may be difficult to know the bound
void init_2d_array(float mat[][4], /* error, arrays of unknown bound */
                   size_t rows, size_t cols)
{
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            mat[r][c] = r * c;
}

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. basic array in C++.
    ------------------------------- */
    int num_array1[5];                   /* uninitialized array, random values */
    int num_array2[5] = {0, 1, 2, 3, 4}; /* initialization */

    for (size_t i = 0; i < 5; i++)
    {
        std::cout << num_array1[i] << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < 5; i++)
    {
        std::cout << num_array2[i] << " ";
    }
    std::cout << std::endl
              << std::endl;

    /* Step 2. const array in C++.
    ------------------------------- */
    const float values[4] = {1.1f, 2.2f, 3.3f, 4.4f};
    // values[0] = 3.14f;

    float sum = array_sum(values, 4);
    std::cout << "the sum of array: " << sum << std::endl
              << std::endl;

    /* Step 3. array index-bound in C++.
    --------------------------------------- */
    int idx_bound_arr[5]{1, 2, 3, 4, 5};
    // for (int i = -1; i <= 5; i++) /* error */
    for (int i = 0; i < 5; i++)
    {
        idx_bound_arr[i] = i * i;
    }
    // for (int i = -1; i <= 5; i++) /* error */
    for (int i = 0; i < 5; i++)
    {
        std::cout << "value in array[" << i << "] = "
                  << idx_bound_arr[i] << std::endl;
    }
    std::cout << std::endl;

    /* Step 4. char array in C++.
    ------------------------------- */
    char rabbit[16] = {'P', 'e', 't', 'e', 'r'};
    std::cout << "String length is " << strlen(rabbit) << std::endl;
    for (int i = 0; i < 16; i++)
        std::cout << i << ":" << +rabbit[i]
                  << "(" << rabbit[i] << ")" << std::endl;

    char bad_pig[9] = {'P', 'e', 'p', 'p', 'a', ' ', 'P', 'i', 'g'};
    // '\0' is the ending char for char array in C and C-style string.
    char good_pig[10] = {'P', 'e', 'p', 'p', 'a', ' ', 'P', 'i', 'g', '\0'};

    std::cout << "Rabbit is (" << rabbit << ")" << std::endl;
    std::cout << "Pig's bad name is (" << bad_pig << ")" << std::endl;
    std::cout << "Pig's good name is (" << good_pig << ")" << std::endl;

    char name[10] = {'L', 'i', '\0', 'W', 'e', '0'};
    std::cout << strlen(name) << std::endl;
    std::cout << name << std::endl
              << std::endl;

    /* Step 5. array must be know the size in C++.
    ----------------------------------------------- */
    int mat1[2][3] = {{11, 12, 13}, {14, 15, 16}};

    int rows = 5;
    int cols = 4;
    // float mat2[rows][cols]; //uninitialized array
    float mat2[rows][4]; /* uninitialized array */

    init_2d_array(mat2, rows, cols);

    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            mat2[r][c] = r * c;

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
            std::cout << mat2[r][c] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl
              << std::endl;

    /* Step 6. char array operators.
    --------------------------------- */
    char str1[] = "Hello, \0CPP";
    char str2[] = "SUSTech";
    char result[128];

    for (int i = 0; i < 16; i++)
        std::cout << i << ":" << +str1[i]
                  << "(" << str1[i] << ")" << std::endl;

    strcpy(result, str1);
    std::cout << "Result = " << result << std::endl;
    strcat(result, str2);
    std::cout << "Result = " << result << std::endl;

    std::cout << "strcmp() = " << strcmp(str1, str2) << std::endl;

    // strcat(str1, str2); /* danger operation! */
    // std::cout << "str1 = " << str1 << std::endl;

    /* Step 7. variable-length array.
    ------------------------------------ */
    // fixed length array, initialized to {0,1,0,0,0}
    int num_array_const[5] = {0, 1};
    std::cout << "sizeof(num_array_const) = "
              << sizeof(num_array_const) << std::endl;

    int len = 0;
    while (len < 10)
    {
        int num_array_var[len]; // variable-length array
        std::cout << "len = " << len;
        std::cout << ", sizeof(num_array_var)) = "
                  << sizeof(num_array_var) << std::endl;
        len++;
    }

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ array.cpp
 * $ clang++ array.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */