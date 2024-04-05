#include "Math.hpp"

#include <cstdlib>

bool MyMath::addNumer(const int &val1, const int &val2, int &res)
{
    res = val1 + val2;
    return true;
}

bool MyMath::subNumer(const int &val1, const int &val2, int &res)
{
    res = val1 - val2;

    return true;
}

bool MyMath::mulNumer(const int &val1, const int &val2, int &res)
{
    res = val1 * val2;

    return true;
}

/**
 * @brief float，double分别遵循R32-24,R64-53的标准,
 * 所以 float的精度误差在1e-6；double精度误差在1e-15，
 * 所以要判断一个单精度浮点数：则是 if( abs(f) <= 1e-6),
 * 判断一个双精度浮点数：则是if( abs(f) <= 1e-15 )；
 * 若小于，为0，大于，不为0 。
 * 
 */
bool MyMath::divNumer(const int &val1, const int &val2, int &res)
{
    bool flag = false;

    if (0 == val2 || std::abs(val2) <= 1e-6 || std::abs(val2) <= 1e-15)
        flag = false;
    else
        res = val1 / val2;

    return flag;
}
