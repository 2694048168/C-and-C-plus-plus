#include "Math/Math.h"

namespace Math {

int MathExample::Add(int a, int b)
{
    return a + b;
}

int MathExample::Sub(int a, int b)
{
    return a - b;
}

int MathExample::Div(int a, int b)
{
    if (0 == b)
        return 0;
    return a / b;
}

int MathExample::Mul(int a, int b)
{
    return a * b;
}

} // namespace Math