/**
 * @file 09_firstExercises.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-21
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief 
 * 1. 创建 enum class Operation, 其值为 Add, Subtract, Multiply 和 Divide
 * 2. 创建 struct Calculator, 应该有一个接受 Operation 的构造函数.
 * 3. 在 Calculator 上创建 int calculate(int a,int b)方法,
 *    在被调用时, 该方法应根据其构造函数参数执行加法、减法、乘法或除法运算, 并返回结果.
 * 4. 尝试用不同的方法来初始化 Calculator 实例.
 * 
 */
enum class Operation : int
{
    Add = 0,
    Subtract,
    Multiply,
    Divide,
    NUM_Operation
};

struct Calculator
{
    Calculator()
    {
        printf("the operator type default\n");
    }

    int calculate(int a, int b)
    {
        switch (m_type)
        {
        case Operation::Add:
        {
            return a + b;
        }
        break;

        case Operation::Subtract:
        {
            return a - b;
        }
        break;

        case Operation::Multiply:
        {
            return a * b;
        }
        break;

        case Operation::Divide:
        {
            if (b == 0)
                return 0;
            return a / b;
        }
        break;

        default:
        {
            printf("NOT Implemented\n");
            return 0;
        }
        }
    }

    Calculator(const Operation &type)
    {
        m_type = type;
        printf("the operator type: %d\n", m_type);
    }

private:
    Operation m_type;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("the Operation::Add value: %d\n", Operation::Add);
    printf("the Operation::Subtract value: %d\n", Operation::Subtract);
    printf("the Operation::Multiply value: %d\n", Operation::Multiply);
    printf("the Operation::Divide value: %d\n", Operation::Divide);
    printf("the number of Operation enum: %d\n", Operation::NUM_Operation);

    Calculator calc;
    Calculator calc_{Operation::Multiply};
    Calculator calc_add{Operation::Add};
    Calculator calc_div{Operation::Divide};
    Calculator calc_sub{Operation::Subtract};

    printf("the result is %d\n", calc.calculate(2, 1));
    printf("the result is %d\n", calc_.calculate(21, 1));
    printf("the result is %d\n", calc_add.calculate(21, 1));
    printf("the result is %d\n", calc_sub.calculate(21, 1));
    printf("the result is %d\n", calc_div.calculate(21, 2));

    return 0;
}
