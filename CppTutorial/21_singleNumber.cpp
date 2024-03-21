/**
 * @file 21_singleNumber.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 寻找一个非空数组中的只出现过一次的数字
 * @version 0.1
 * @date 2024-03-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <vector>


/**
 * @brief 一个非空整数数组, 除了某个元素只出现一次以外,
 * 其余每个元素均出现两次, 找出那个只出现了一次的元素.
 *  
 * ======== Solution ======== 
 * step 1. 以需求为导向, 需要的是只出现过一次的数字, 即需要消除出现过两次的数字;
 * step 2. 考虑一下数值可以进行的操作, 通过异或运算;
 * step 3. 异或运算导致两个相同数字结果为 0;
 * step 4. 异或运算导致任何数字与 0结果为该数字;
 * step 5. 好像可以满足需求, 不过需要进一步确定异或运算是否对顺序敏感;
 * step 6. search 一下, 异或运算满足交换律和结合律;
 * step 7. nice, 做一下简单的数学推导论证一下;
 * 
 * ======== Proof ========
 * 
 * 假设数组中有 2m+1 个数, 其中有 m 个数各出现两次, 一个数出现一次;
 * 令 a_{1} ~ a_{m} 为出现两次的 m 个数, a_{m+1} 为出现一次的数;
 * 根据 step 6. 数组中的全部元素的异或运算结果如下:
 * result = (a1⊕a1)⊕(a2⊕a2)⊕⋯⊕(am⊕am)⊕a_{m+1}
 *        = 0 ⊕ 0 ⊕ 0 ⊕⋯⊕ 0 ⊕ a_{m+1}
 *        = a_{m+1}
 *
 * So, 数组中的全部元素的异或运算结果即为数组中只出现一次的数字
 * 
 * @param vec 
 * @return int 
 */
int singleNumber(const std::vector<int> &vec)
{
    int res = 0;
    for (const auto &elem : vec)
    {
        res ^= elem;
    }

    return res;
}

// ===================================
int main(int argc, const char **argv)
{
    std::vector<int> numVec{2, 1, 2, 1, 3, 42, 12, 12, 45, 54, 66, 45, 66, 54};

    std::cout << "The only one number: " << singleNumber(numVec) << '\n';

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\21_singleNumber.cpp -std=c++23
// g++ .\21_singleNumber.cpp -std=c++23
