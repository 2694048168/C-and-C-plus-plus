/**
 * @file StrategyUML.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-01
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ StrategyUML.cpp -std=c++20
 * clang++ StrategyUML.cpp -std=c++20
 * 
 */

#include <iostream>
#include <vector>

// Demo1: 根据不同的策略, 计算出不同的结果
// Demo2: 根据不同的策略, 打印不同的提示语
class Strategy
{
public:
    virtual int  operation(int input) const = 0;
    virtual void execute() const            = 0;

    virtual ~Strategy() = default;
};

class ConcreteStrategyA : public Strategy
{
public:
    int operation(int input) const override
    {
        return input * 2;
    }

    virtual void execute() const override
    {
        std::cout << "Using Strategy A.\n";
    }
};

class ConcreteStrategyB : public Strategy
{
public:
    int operation(int input) const override
    {
        return input / 2;
    }

    virtual void execute() const override
    {
        std::cout << "Using Strategy B.\n";
    }
};

class Context
{
private:
    Strategy *strategy;

public:
    Context(Strategy *strategy = nullptr)
    {
        this->strategy = strategy;
    }

    void setStrategy(Strategy *strategy)
    {
        this->strategy = strategy;
    }

    int execute(int input) const
    {
        return strategy->operation(input);
    }

    void executeStrategy() const
    {
        strategy->execute();
    }
};

// Demo: 集成冒泡排序算法、选择排序算法给客户端进行调用
class SortingStrategy
{
public:
    virtual void sort(std::vector<int> &arr) = 0;
};

class BubbleSort : public SortingStrategy
{
public:
    void sort(std::vector<int> &arr) override
    {
        std::cout << "\nUse Strategy: BubbleSort." << std::endl;
        int n = arr.size();
        for (size_t i = 0; i < n - 1; ++i)
        {
            for (size_t j = 0; j < n - i - 1; ++j)
            {
                if (arr[j] > arr[j + 1])
                {
                    std::swap(arr[j], arr[j + 1]);
                }
            }
        }
    }
};

class SelectionSort : public SortingStrategy
{
public:
    void sort(std::vector<int> &arr) override
    {
        std::cout << "\nUse Strategy: SelectionSort." << std::endl;
        int n = arr.size();
        for (int i = 0; i < n - 1; ++i)
        {
            int minIndex = i;
            for (int j = i + 1; j < n; ++j)
            {
                if (arr[j] < arr[minIndex])
                {
                    minIndex = j;
                }
            }
            std::swap(arr[i], arr[minIndex]);
        }
    }
};

class SortContext
{
private:
    SortingStrategy *strategy;

public:
    void setStrategy(SortingStrategy *strategy)
    {
        this->strategy = strategy;
    }

    void executeStrategy(std::vector<int> &arr)
    {
        strategy->sort(arr);
    }
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    Context context(new ConcreteStrategyA());
    std::cout << "Using Strategy A: " << context.execute(10) << std::endl;
    context.setStrategy(new ConcreteStrategyB());
    std::cout << "Using Strategy B: " << context.execute(10) << std::endl;

    Strategy *strategyA = new ConcreteStrategyA();
    Strategy *strategyB = new ConcreteStrategyB();

    Context context_(strategyA);
    context_.executeStrategy();
    context_.setStrategy(strategyB);
    context_.executeStrategy();

    delete strategyA;
    delete strategyB;

    std::cout << "--------------------------------------\n";
    std::vector<int> data = {23, 5, 6, 36, 25, 4, 20};

    SortContext   context_sort;
    BubbleSort    bubbleSort;
    SelectionSort selectSort;

    context_sort.setStrategy(&bubbleSort);
    context_sort.executeStrategy(data);

    for (const auto &num : data)
    {
        std::cout << num << " ";
    }

    data = {32, 45, 5, 6, 100, 7};
    context_sort.setStrategy(&selectSort);
    context_sort.executeStrategy(data);

    for (const auto &num : data)
    {
        std::cout << num << " ";
    }

    return 0;
}
