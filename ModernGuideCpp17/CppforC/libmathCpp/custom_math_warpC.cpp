#include "custom_math_warpC.h"

#include "custom_math.h"

struct CustomMathWrapper
{
    CustomMath *instance;
};

extern "C"
{
// 构造函数
pCustomMathWrapper CUSTOM_MATH_API __callback__ CustomMath_create()
{
    return new CustomMathWrapper{new CustomMath};
}

// 析构函数
void CUSTOM_MATH_API __callback__ CustomMath_destroy(CustomMathWrapper *wrapper)
{
    delete wrapper->instance;
    delete wrapper;
}

// 公有函数
int CUSTOM_MATH_API __callback__ CustomMath_add(CustomMathWrapper *wrapper, int x, int y)
{
    return wrapper->instance->custom_add(x, y);
}

int CUSTOM_MATH_API __callback__ CustomMath_sub(CustomMathWrapper *wrapper, int x, int y)
{
    return wrapper->instance->custom_sub(x, y);
}

int CUSTOM_MATH_API __callback__ CustomMath_mul(CustomMathWrapper *wrapper, int x, int y)
{
    return wrapper->instance->custom_mul(x, y);
}

int CUSTOM_MATH_API __callback__ CustomMath_div(CustomMathWrapper *wrapper, int x, int y)
{
    return wrapper->instance->custom_div(x, y);
}
}
