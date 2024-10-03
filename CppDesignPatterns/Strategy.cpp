/**
 * @file Strategy.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

class Context
{
};

// 稳定点: 抽象去解决它
// 变化点: 扩展(继承和组合)去解决它
class ProStrategy
{
public:
    virtual double CalcPro(const Context &ctx) = 0;

    virtual ~ProStrategy() {}
};

class VAC_Spring : public ProStrategy
{
    virtual double CalcPro(const Context &ctx) override
    {
        std::cout << "VAC_Spring : public ProStrategy\n";
        return 0.;
    }
};

class VAC_GuoQin : public ProStrategy
{
    virtual double CalcPro(const Context &ctx) override
    {
        std::cout << "VAC_GuoQin : public ProStrategy\n";
        return 0.;
    }
};

class VAC_QiXi : public ProStrategy
{
    virtual double CalcPro(const Context &ctx) override
    {
        std::cout << "VAC_QiXi : public ProStrategy\n";
        return 0.;
    }
};

// 设计原则: 接口隔离原则;
// 两种方法: 1. 采用具体接口选择算法; 2. 依赖注入
class Promotion
{
public:
    Promotion(ProStrategy *pStrategy = nullptr)
        : m_pStrategy(pStrategy)
    {
    }

    ~Promotion() {}

    void Choose(ProStrategy *pStrategy)
    {
        if (pStrategy)
            m_pStrategy = pStrategy;
    }

    double CalcPromotion(const Context &ctx)
    {
        if (m_pStrategy)
            return m_pStrategy->CalcPro(ctx);
        return 0.0L;
    }

private:
    ProStrategy *m_pStrategy;
};

// ====================================
int main(int argc, const char **argv)
{
    Context      ctx;
    ProStrategy *pStrategy  = new VAC_GuoQin();
    Promotion   *pPromotion = new Promotion(pStrategy);

    pPromotion->CalcPromotion(ctx);
    pPromotion->Choose(new VAC_QiXi());
    pPromotion->CalcPromotion(ctx);

    if (pStrategy)
    {
        delete pStrategy;
        pStrategy = nullptr;
    }

    if (pPromotion)
    {
        delete pPromotion;
        pPromotion = nullptr;
    }

    return 0;
}
