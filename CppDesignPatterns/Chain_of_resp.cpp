/**
 * @file Chain_of_resp.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

struct Context
{
    std::string name = "";
    int         day  = 0;
};

// 稳定点 抽象; 变化点 扩展(多态)
// 从单个处理节点出发, 能够处理结束, 则处理结束; 否则继续下一个处理节点;
// 链表关系如何抽象
class InterfaceHandler
{
public:
    virtual ~InterfaceHandler()
    {
        if (m_pNext)
        {
            delete m_pNext;
            m_pNext = nullptr;
        }
    }

    void SetNextHandler(InterfaceHandler *pNext)
    {
        // 链表关系
        m_pNext = pNext;
    }

    bool Handle(const Context &ctx)
    {
        if (CanHandle(ctx))
        {
            return HandleRequest(ctx);
        }
        else if (GetNextHandler())
        {
            return GetNextHandler()->Handle(ctx);
        }
        else
        {
            // error
            return false;
        }
        return false;
    }

    static bool handler_leave_req(Context &ctx);

protected:
    virtual bool HandleRequest(const Context &ctx)
    {
        return true;
    }

    virtual bool CanHandle(const Context &ctx)
    {
        return true;
    }

    InterfaceHandler *GetNextHandler()
    {
        return m_pNext;
    }

private:
    InterfaceHandler *m_pNext; // 组合基类指针
};

// ========== 具体任务处理节点 ==========
class HandlerByBeauty : public InterfaceHandler
{
protected:
    virtual bool HandleRequest(const Context &ctx) override
    {
        return true;
    }

    virtual bool CanHandle(const Context &ctx) override
    {
        if (ctx.day <= 3)
            return true;

        return false;
    }
};

class HandlerByMainProgram : public InterfaceHandler
{
protected:
    virtual bool HandleRequest(const Context &ctx) override
    {
        return true;
    }

    virtual bool CanHandle(const Context &ctx) override
    {
        if (ctx.day <= 5)
            return true;

        return false;
    }
};

class HandlerByLeader : public InterfaceHandler
{
protected:
    virtual bool HandleRequest(const Context &ctx) override
    {
        return true;
    }

    virtual bool CanHandle(const Context &ctx) override
    {
        if (ctx.day <= 10)
            return true;

        return false;
    }
};

class HandlerByBoos : public InterfaceHandler
{
protected:
    virtual bool HandleRequest(const Context &ctx) override
    {
        return true;
    }

    virtual bool CanHandle(const Context &ctx) override
    {
        if (ctx.day <= 15)
            return true;

        return false;
    }
};

bool InterfaceHandler::handler_leave_req(Context &ctx)
{
    InterfaceHandler *pNode_0 = new HandlerByBeauty();
    InterfaceHandler *pNode_1 = new HandlerByMainProgram();
    InterfaceHandler *pNode_2 = new HandlerByLeader();
    InterfaceHandler *pNode_3 = new HandlerByBoos();

    pNode_0->SetNextHandler(pNode_1);
    pNode_1->SetNextHandler(pNode_2);
    pNode_2->SetNextHandler(pNode_3);
    return pNode_0->Handle(ctx);
}

// -----------------------------------
int main(int argc, const char **argv)
{
    Context ctx{"Ithaca", 15};

    bool flag = InterfaceHandler::handler_leave_req(ctx);
    if (flag)
    {
        std::cout << "[Successfully] ---> " << ctx.name << " All Node is OK\n";
    }
    else
    {
        std::cout << "[Failed] ---> " << ctx.name << " All Node is NG\n";
    }

    Context ctx_{"WeiLi", 25};

    bool flag_ = InterfaceHandler::handler_leave_req(ctx_);
    if (flag_)
    {
        std::cout << "[Successfully] ---> " << ctx_.name << " All Node is OK\n";
    }
    else
    {
        std::cout << "[Failed] ---> " << ctx_.name << " All Node is NG\n";
    }

    return 0;
}
