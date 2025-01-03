/**
 * @file Interpreter.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-03
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Interpreter.cpp -std=c++20
 * clang++ Interpreter.cpp -std=c++20
 * 
 */

#include <iostream>
#include <map>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

// Demo1: 解析数学表达式
class Expression
{
public:
    virtual int interpret() = 0;

    virtual ~Expression() = default;
};

class NumberExpression : public Expression
{
public:
    NumberExpression(int number)
        : m_number(number)
    {
    }

    int interpret() override
    {
        return m_number;
    }

private:
    int m_number;
};

class AddExpression : public Expression
{
public:
    AddExpression(Expression *left, Expression *right)
        : m_left(left)
        , m_right(right)
    {
    }

    int interpret() override
    {
        return m_left->interpret() + m_right->interpret();
    }

private:
    Expression *m_left;
    Expression *m_right;
};

class Context
{
public:
    Expression *evaluate(const std::string &expression)
    {
        std::stack<Expression *> stack;
        std::stringstream        ss(expression);
        std::string              token;
        while (getline(ss, token, ' '))
        {
            if (token == "+")
            {
                Expression *right = stack.top();
                stack.pop();
                Expression *left = stack.top();
                stack.pop();
                Expression *addExp = new AddExpression(left, right);
                stack.push(addExp);
            }
            else
            {
                int         number    = stoi(token);
                Expression *numberExp = new NumberExpression(number);
                stack.push(numberExp);
            }
        }
        return stack.top();
    }
};

// Demo2: 解析字符串
class StrExpression
{
public:
    virtual bool interpret(const std::string &context) = 0;

    virtual ~StrExpression() = default;
};

class TerminalExpression : public StrExpression
{
private:
    std::string expression;

public:
    TerminalExpression(const std::string &expression)
        : expression(expression)
    {
    }

    bool interpret(const std::string &context) override
    {
        if (context.find(expression) != std::string::npos)
        {
            return true;
        }
        return false;
    }
};

class OrExpression : public StrExpression
{
private:
    StrExpression *expr1;
    StrExpression *expr2;

public:
    OrExpression(StrExpression *expr1, StrExpression *expr2)
        : expr1(expr1)
        , expr2(expr2)
    {
    }

    bool interpret(const std::string &context) override
    {
        return expr1->interpret(context) || expr2->interpret(context);
    }
};

// Demo1: 模拟的科学计算库
class SciExpression
{
public:
    virtual double evaluate() = 0;

    virtual ~SciExpression() = default;
};

class Addition : public SciExpression
{
private:
    double left;
    double right;

public:
    Addition(double l, double r)
        : left(l)
        , right(r)
    {
    }

    double evaluate() override
    {
        return left + right;
    }
};

class Subtraction : public SciExpression
{
private:
    double left;
    double right;

public:
    Subtraction(double l, double r)
        : left(l)
        , right(r)
    {
    }

    double evaluate() override
    {
        return left - right;
    }
};

class Interpreter
{
private:
    SciExpression *expr;

public:
    Interpreter(SciExpression *e)
        : expr(e)
    {
    }

    double interpret()
    {
        return expr->evaluate();
    }
};

// Demo2: 模拟的汇编语言解析器
class Instruction
{
public:
    virtual void execute() const = 0;
};

class LoadInstruction : public Instruction
{
private:
    int value;

public:
    LoadInstruction(int v)
        : value(v)
    {
    }

    void execute() const override
    {
        std::cout << "Loading value: " << value << std::endl;
    }
};

class StoreInstruction : public Instruction
{
private:
    int address;

public:
    StoreInstruction(int addr)
        : address(addr)
    {
    }

    void execute() const override
    {
        std::cout << "Storing at address: " << address << std::endl;
    }
};

class InstructionInterpreter
{
private:
    std::vector<Instruction *> instructions;

public:
    void addInstruction(Instruction *instr)
    {
        instructions.push_back(instr);
    }

    void interpret()
    {
        for (auto &instr : instructions)
        {
            instr->execute();
        }
    }
};

// Demo3: 模拟的shell脚本解析器
class Command
{
public:
    virtual ~Command() {}

    virtual void execute() = 0;
};

class CDCommand : public Command
{
private:
    std::string directory;

public:
    CDCommand(const std::string &dir)
        : directory(dir)
    {
    }

    void execute() override
    {
        std::cout << "Changing directory to " << directory << std::endl;
    }
};

class LSCommand : public Command
{
public:
    void execute() override
    {
        std::cout << "Listing files in current directory\n";
    }
};

class EchoCommand : public Command
{
private:
    std::string message;

public:
    EchoCommand(const std::string &msg)
        : message(msg)
    {
    }

    void execute() override
    {
        std::cout << "Echoing: " << message << "\n";
    }
};

class ShellInterpreter
{
private:
    std::map<std::string, Command *> command_map;

public:
    void add_command(const std::string &name, Command *cmd)
    {
        command_map[name] = cmd;
    }

    void interpret(const std::string &cmd_str)
    {
        if (command_map.find(cmd_str) != command_map.end())
        {
            command_map[cmd_str]->execute();
        }
        else
        {
            std::cerr << "Unknown command: " << cmd_str << '\n';
        }
    }
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    Context context;

    Expression *expression = context.evaluate("2 3 +");
    int         result     = expression->interpret();
    std::cout << "Result: " << result << std::endl;

    delete expression;

    std::cout << "--------------------------------------\n";
    StrExpression *expr1  = new TerminalExpression("hello");
    StrExpression *expr2  = new TerminalExpression("world");
    StrExpression *orExpr = new OrExpression(expr1, expr2);

    std::vector<std::string> contexts = {"hello", "world", "hello world", "hi"};

    for (const auto &context : contexts)
    {
        if (orExpr->interpret(context))
        {
            std::cout << "Expression is true for context: " << context << std::endl;
        }
        else
        {
            std::cout << "Expression is false for context: " << context << std::endl;
        }
    }

    delete expr1;
    delete expr2;
    delete orExpr;

    std::cout << "--------------------------------------\n";
    SciExpression *addExpr = new Addition(3.0, 5.0);
    SciExpression *subExpr = new Subtraction(10.0, 7.0);

    Interpreter intp(addExpr);
    std::cout << "Addition result: " << intp.interpret() << std::endl;
    Interpreter intp2(subExpr);
    std::cout << "Subtraction result: " << intp2.interpret() << std::endl;

    delete addExpr;
    delete subExpr;

    std::cout << "--------------------------------------\n";
    InstructionInterpreter interpreter;

    LoadInstruction  loadInstr(5);
    StoreInstruction storeInstr(10);

    interpreter.addInstruction(&loadInstr);
    interpreter.addInstruction(&storeInstr);

    interpreter.interpret();

    std::cout << "--------------------------------------\n";
    ShellInterpreter interpreter_;

    interpreter_.add_command("cd", new CDCommand("/path/to/directory"));
    interpreter_.add_command("ls", new LSCommand());
    interpreter_.add_command("echo", new EchoCommand("Hello, World!"));

    interpreter_.interpret("cd /home");
    interpreter_.interpret("ls");
    interpreter_.interpret("echo hi");

    return 0;
}
