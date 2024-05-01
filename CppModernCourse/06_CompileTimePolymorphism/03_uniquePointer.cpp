/**
 * @file 03_uniquePointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>
#include <utility>

/**
 * @brief 唯一指针(unique pointer)是一个针对自由存储区分配的对象的 RAII 包装器.
 * 顾名思义, 唯一指针在同一时间只有一个所有者, 
 * 所以当唯一指针的生命周期结束时, 指向的对象会被销毁.
 * 
 * SimpleUniquePointer 是标准库中 std::unique_ptr 的教学实现,
 * 它是 RAII 模板家族的成员, 称为智能指针(smart pointers).
 * 
 */
template<typename T>
class SimpleUniquePointer
{
public:
    SimpleUniquePointer() = default;

    SimpleUniquePointer(T *pointer)
        : m_pointer{pointer}
    {
    }

    ~SimpleUniquePointer()
    {
        if (m_pointer)
        {
            delete m_pointer;
            m_pointer = nullptr;
        }
    }

    // 想被管理对象只有单一所有者, 所以使用 delete 删除了复制构造函数和复制赋值运算符
    SimpleUniquePointer(const SimpleUniquePointer &)            = delete;
    SimpleUniquePointer &operator=(const SimpleUniquePointer &) = delete;

    SimpleUniquePointer(SimpleUniquePointer &&other) noexcept
        : m_pointer{other.m_pointer}
    {
        other.m_pointer = nullptr;
    }

    SimpleUniquePointer &operator=(SimpleUniquePointer &&other) noexcept
    {
        if (m_pointer)
            delete m_pointer;
        m_pointer     = other.m_pointer;
        other.pointer = nullptr;
        return *this;
    }

    T *get() const
    {
        return m_pointer;
    }

private:
    T *m_pointer;
};

struct Tracer
{
    Tracer(const char *name)
        : name{name}
    {
        printf("%s constructed.\n", name);
    }

    ~Tracer()
    {
        printf("%s destructed.\n", name);
    }

private:
    const char *const name;
};

void consumer(SimpleUniquePointer<Tracer> consumer_ptr)
{
    printf("(cons) consumer_ptr: 0x%p\n", consumer_ptr.get());
}

// ----------------------------------
int main(int argc, const char **argv)
{
    auto ptr_a = SimpleUniquePointer(new Tracer{"ptr_a"});
    printf("(main) ptr_a: 0x%p\n", ptr_a.get());
    consumer(std::move(ptr_a));
    printf("(main) ptr_a: 0x%p\n", ptr_a.get());

    return 0;
}
