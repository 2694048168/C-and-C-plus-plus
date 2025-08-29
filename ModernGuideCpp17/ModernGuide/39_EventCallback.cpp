/**
 * @file 39_EventCallback.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-29
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <iostream>
#include <string>

/**
 * @brief 在C++中, 想要实现事件的派发需要绑定函数指针来进行事件回调派发.
 * Modern C++17中, 增加了类模板 std::function, 是通用的多态函数包装器.
 * 有了它, 灵活性大大提高, 通过简单的封装, 便可以实现便于扩展的Event事件回调机制.
 * 
 */
#include <functional>
#include <unordered_map>

namespace Ithaca {

template<typename... ArgTypes>
class Event
{
public:
    using Callback = std::function<void(ArgTypes...)>;

    uint64_t AddCallback(Callback callback)
    {
        uint64_t id = m_addIdCount++;
        m_callbacks.emplace(id, callback);
        return id;
    }

    uint64_t operator+=(Callback callback)
    {
        return AddCallback(callback);
    }

    bool RemoveCallback(uint64_t id)
    {
        return m_callbacks.erase(id) != 0;
    }

    bool operator-=(uint64_t id)
    {
        return RemoveCallback(id);
    }

    void RemoveAllCallbacks()
    {
        m_callbacks.clear();
    }

    uint64_t GetCallbackCount()
    {
        return m_callbacks.size();
    }

    void Invoke(ArgTypes... args)
    {
        for (const auto &[key, value] : m_callbacks)
        {
            value(args...);
        }
    }

private:
    std::unordered_map<uint64_t, Callback> m_callbacks;

    uint64_t m_addIdCount = 0;
};

} // namespace Ithaca

void Print(const std::string &name, int age)
{
    std::cout << "Print:    " << "name: " << name << ", age: " << age << std::endl;
}

class Test
{
public:
    Ithaca::Event<const std::string &, int> m_peopleEvent;

    void Print(const std::string &name, int age)
    {
        std::cout << "Test::Print:    " << "name: " << name << ", age: " << age << std::endl;
        m_peopleEvent.Invoke(name, age);
    }
};

// ----------------------------------
int main(int argc, const char **argv)
{
    Test test;
    std::cout << "注册Event前打印" << std::endl;
    test.Print("小明", 14);

    uint64_t callbackId = test.m_peopleEvent.AddCallback(Print);
    std::cout << "注册Event后打印" << std::endl;
    test.Print("小红", 13);

    std::cout << "删除Event注册后打印" << std::endl;
    test.m_peopleEvent.RemoveCallback(callbackId);
    test.Print("小美", 14);

    return 0;
}
