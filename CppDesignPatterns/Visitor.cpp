/**
 * @file Visitor.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-03
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Visitor.cpp -std=c++20
 * clang++ Visitor.cpp -std=c++20
 * 
 */

#include <iostream>
#include <vector>

class ElementA;
class ElementB;

class Visitor
{
public:
    virtual void visit(ElementA *element) = 0;
    virtual void visit(ElementB *element) = 0;
};

class Element
{
public:
    virtual void accept(Visitor *visitor) = 0;
};

class ElementA : public Element
{
public:
    void accept(Visitor *visitor) override
    {
        visitor->visit(this);
    }

    void operationA()
    {
        std::cout << "OperationA called on ElementA." << std::endl;
    }
};

class ElementB : public Element
{
public:
    void accept(Visitor *visitor) override
    {
        visitor->visit(this);
    }

    void operationB()
    {
        std::cout << "OperationB called on ElementB." << std::endl;
    }
};

class ConcreteVisitor : public Visitor
{
public:
    void visit(ElementA *element) override
    {
        std::cout << "visits ElementA." << std::endl;
        element->operationA();
    }

    void visit(ElementB *element) override
    {
        std::cout << "visits ElementB." << std::endl;
        element->operationB();
    }
};

// Demo1: 模拟的XML解析器
class ElementXML;
class Text;

class VisitorXML
{
public:
    virtual void visit(ElementXML &element) = 0;
    virtual void visit(Text &text)          = 0;
};

class Node
{
public:
    virtual void accept(VisitorXML &visitor) = 0;
};

class ElementXML : public Node
{
public:
    std::string name;

    ElementXML(const std::string &n)
        : name(n)
    {
    }

    void accept(VisitorXML &visitor) override
    {
        visitor.visit(*this);
    }
};

class Text : public Node
{
public:
    std::string content;

    Text(const std::string &c)
        : content(c)
    {
    }

    void accept(VisitorXML &visitor) override
    {
        visitor.visit(*this);
    }
};

class XmlDocument
{
public:
    std::vector<Node *> nodes;

    void addNode(Node *node)
    {
        nodes.push_back(node);
    }

    void accept(VisitorXML &visitor)
    {
        for (auto node : nodes)
        {
            node->accept(visitor);
        }
    }
};

class ElementCounter : public VisitorXML
{
public:
    int elementCount = 0;

    void visit(ElementXML &element) override
    {
        std::cout << "Found element: " << element.name << std::endl;
        elementCount++;
    }

    void visit(Text &text) override
    {
        std::cout << "Found text: " << text.content << std::endl;
    }
};

// Demo2: 模拟的AST抽象语法树
class Number;
class BinaryOperation;

class ExpressionVisitor
{
public:
    virtual double visit(Number &number)                   = 0;
    virtual double visit(BinaryOperation &binaryOperation) = 0;
};

class Expression
{
public:
    virtual double accept(ExpressionVisitor &visitor) = 0;
};

class Number : public Expression
{
public:
    double value;

    Number(double v)
        : value(v)
    {
    }

    double accept(ExpressionVisitor &visitor) override
    {
        return visitor.visit(*this);
    }
};

class BinaryOperation : public Expression
{
public:
    char        operation;
    Expression *left;
    Expression *right;

    BinaryOperation(char op, Expression *l, Expression *r)
        : operation(op)
        , left(l)
        , right(r)
    {
    }

    double accept(ExpressionVisitor &visitor) override
    {
        return visitor.visit(*this);
    }
};

class ExpressionEvaluator : public ExpressionVisitor
{
public:
    double visit(Number &number) override
    {
        return number.value;
    }

    double visit(BinaryOperation &binaryOperation) override
    {
        double left  = binaryOperation.left->accept(*this);
        double right = binaryOperation.right->accept(*this);
        switch (binaryOperation.operation)
        {
        case '+':
            return left + right;
        case '-':
            return left - right;
        case '*':
            return left * right;
        case '/':
            if (right != 0)
            {
                return left / right;
            }
            else
            {
                std::cerr << "Division by zero" << std::endl;
                return 0;
            }
        default:
            std::cerr << "Invalid operation: " << binaryOperation.operation << std::endl;
            return 0;
        }
    }
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    ElementA elementA;
    ElementB elementB;

    ConcreteVisitor visitor;

    Element *elements[] = {&elementA, &elementB};
    for (Element *element : elements)
    {
        element->accept(&visitor);
    }

    std::cout << "--------------------------------------\n";
    XmlDocument document;
    ElementXML  element1("div");

    Text       text("Hello, world!");
    ElementXML element2("p");

    document.addNode(&element1);
    document.addNode(&text);
    document.addNode(&element2);

    ElementCounter counter;
    document.accept(counter);

    std::cout << "Total elements found: " << counter.elementCount << std::endl;

    std::cout << "--------------------------------------\n";
    Number num1(5.0);
    Number num2(3.0);
    Number num3(2.0);

    BinaryOperation expr1('+', &num1, &num2);
    BinaryOperation expr2('*', &expr1, &num3);

    ExpressionEvaluator evaluator;

    double result = expr2.accept(evaluator);
    std::cout << "Result: " << result << std::endl;

    return 0;
}
