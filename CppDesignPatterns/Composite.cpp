/**
 * @file Composite.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-26
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Composite.cpp -std=c++20
 * clang++ Composite.cpp -std=c++20
 * 
 */

#include <iostream>
#include <vector>

// Demo1: 先操作叶子节点, 后操作主节点
class Component
{
public:
    virtual void operation() const = 0;

    virtual ~Component() {}
};

class Leaf : public Component
{
public:
    Leaf(const std::string &name)
        : name_(name)
    {
    }

    virtual void operation() const override
    {
        std::cout << "Operation on leaf: " << name_ << '\n';
    }

private:
    std::string name_;
};

class Composite : public Component
{
public:
    Composite(const std::string &name)
        : name_(name)
        , children_{}
    {
    }

    void add(Component *component)
    {
        children_.push_back(component);
    }

    void operation() const override
    {
        for (const auto &child : children_)
        {
            child->operation();
        }
        std::cout << "Operation on composite: " << name_ << '\n';
    }

private:
    std::vector<Component *> children_;
    std::string              name_;
};

// Demo2: 先操作主节点, 后操作叶子节点
class Composite_ : public Component
{
public:
    Composite_(const std::string &name)
        : Component()
        , children()
        , _name(name)
    {
    }

    void add(Component *component)
    {
        children.push_back(component);
    }

    void remove(Component *component)
    {
        children.erase(std::remove(children.begin(), children.end(), component), children.end());
    }

    void operation() const override
    {
        std::cout << "Operation on composite: " << _name << '\n';
        for (auto &child : children) child->operation();
    }

private:
    std::vector<Component *> children;
    std::string              _name;
};

// 代码实战: 基于组合模式实现的文件系统
class FileSystemComponent
{
public:
    virtual void display() const   = 0;
    virtual ~FileSystemComponent() = default;
};

class File : public FileSystemComponent
{
public:
    File(const std::string &name, int size)
        : name(name)
        , size(size)
    {
    }

    void display() const override
    {
        std::cout << "File: " << name << " (" << size << " bytes)\n";
    }

private:
    std::string name;
    int         size;
};

class Directory : public FileSystemComponent
{
public:
    Directory(const std::string &name)
        : name(name)
    {
    }

    void display() const override
    {
        std::cout << "Directory: " << name << '\n';
        for (const auto &component : components)
        {
            component->display();
        }
    }

    void addComponent(FileSystemComponent *component)
    {
        components.push_back(component);
    }

private:
    std::string                        name;
    std::vector<FileSystemComponent *> components;
};

// -------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "-----------------------\n";
    Composite root("Composite Root");
    Leaf      leaf1("Leaf 1");
    Leaf      leaf2("Leaf 2");
    Leaf      leaf3("Leaf 3");

    root.add(&leaf1);
    root.add(&leaf2);
    root.add(&leaf3);

    root.operation();

    std::cout << "-----------------------\n";
    Composite_ root_("Composite1");
    root_.add(new Leaf("Leaf1"));
    root_.add(new Leaf("Leaf2"));
    root_.add(new Composite("Composite2"));
    root_.add(new Leaf("Leaf3"));
    root_.operation();

    std::cout << "-----------------------\n";
    FileSystemComponent *file1 = new File("document.txt", 1024);
    FileSystemComponent *file2 = new File("image.jpg", 2048);

    Directory *directory = new Directory("My Documents");

    directory->addComponent(file1);
    directory->addComponent(file2);

    directory->display();

    return 0;
}
