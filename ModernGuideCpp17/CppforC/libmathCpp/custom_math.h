#pragma once

class CustomMath
{
public:
    CustomMath()  = default;
    ~CustomMath() = default;

    int custom_add(int x, int y);
    int custom_sub(int x, int y);
    int custom_mul(int x, int y);
    int custom_div(int x, int y);
};
