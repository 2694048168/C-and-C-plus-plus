/**
 * @file Iterator.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-03
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Iterator.cpp -std=c++20
 * clang++ Iterator.cpp -std=c++20
 * 
 */

#include <algorithm>
#include <iostream>
#include <list>
#include <string>
#include <vector>

// Demo1: 内部迭代器
// Demo2: 外部迭代器
class MyData
{
public:
    MyData(int tmpValue)
    {
        value = tmpValue;
    }

    int value;
};

template<typename T, typename OutIt>
OutIt print_data(const std::list<T> &container, OutIt out)
{
    for (const auto &data : container)
    {
        *out++ = " [";
        *out++ = std::to_string(data.value);
        *out++ = "]";
    }
    return out;
}

class Iterator
{
public:
    virtual bool hasNext() const = 0;
    virtual int  next()          = 0;

    virtual ~Iterator() = default;
};

class ConcreteIterator : public Iterator
{
private:
    std::vector<int> Aggregate;
    int              index = 0;

public:
    ConcreteIterator(const std::vector<int> &vec)
    {
        Aggregate = vec;
    }

    bool hasNext() const override
    {
        return index < Aggregate.size();
    }

    int next() override
    {
        return Aggregate[index++];
    }
};

class Aggregate
{
public:
    virtual Iterator *createIterator() const = 0;
};

class ConcreteAggregate : public Aggregate
{
private:
    std::vector<int> elements;

public:
    ConcreteAggregate(const std::vector<int> &vec)
    {
        elements = vec;
    }

    Iterator *createIterator() const override
    {
        return new ConcreteIterator(elements);
    }
};

void traverseElements(ConcreteAggregate Aggregate)
{
    Iterator *iterator = Aggregate.createIterator();
    while (iterator->hasNext())
    {
        int element = iterator->next();
        std::cout << element << " ";
    }
    std::cout << std::endl;
    delete iterator;
}

// 基于迭代器模式封装的链表
template<typename T>
struct Node
{
    T     data;
    Node *next;
};

template<typename T>
class Iterator_
{
public:
    virtual T    next()    = 0;
    virtual bool hasNext() = 0;

    virtual ~Iterator_() = default;
};

template<typename T>
class LinkedListIterator : public Iterator_<T>
{
public:
    LinkedListIterator(Node<T> *start)
    {
        current_ = start;
    }

    T next() override
    {
        T data   = current_->data;
        current_ = current_->next;
        return data;
    }

    bool hasNext() override
    {
        return current_ != nullptr;
    }

private:
    Node<T> *current_;
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    std::vector<int> vec = {1, 2, 3, 4, 5};
    for (auto it = vec.begin(); it != vec.end(); ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << '\n';

    std::cout << "--------------------------------------\n";
    std::list<MyData> dataList = {10, 20, 30, 40, 50};

    std::ostream_iterator<std::string> outIter(std::cout, "");
    print_data(dataList, outIter);

    std::cout << "\n--------------------------------------\n";
    std::vector<int>  elements = {1, 2, 3, 4, 5};
    ConcreteAggregate Aggregate(elements);
    traverseElements(Aggregate);

    std::cout << "\n--------------------------------------\n";
    Node<int> *head = nullptr;
    for (int i = 6; i >= 1; i--)
    {
        Node<int> *newNode = new Node<int>{i, head};
        head               = newNode;
    }

    Iterator_<int> *iterator = new LinkedListIterator<int>(head);
    while (iterator->hasNext())
    {
        std::cout << iterator->next() << " ";
    }

    while (head != nullptr)
    {
        Node<int> *temp = head;
        head            = head->next;
        delete temp;
    }

    delete iterator;

    return 0;
}
