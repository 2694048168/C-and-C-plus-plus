/**
 * @file ChainResponsibility.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-01
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ ChainResponsibility.cpp -std=c++20
 * clang++ ChainResponsibility.cpp -std=c++20
 * 
 */

#include <iostream>
#include <string>
#include <vector>

class Handler
{
protected:
    Handler *successor;

public:
    void setSuccessor(Handler *successor)
    {
        this->successor = successor;
    }

    virtual ~Handler() = default;

    virtual void handleRequest(const std::string &request) = 0;
};

class ConcreteHandler1 : public Handler
{
public:
    void handleRequest(const std::string &request) override
    {
        if (request == "Type1")
        {
            std::cout << "Handling request of type Type1.\n";
        }
        else if (successor != nullptr)
        {
            successor->handleRequest(request);
        }
        else
        {
            std::cout << "Unable to handle the request.\n";
        }
    }
};

class ConcreteHandler2 : public Handler
{
public:
    void handleRequest(const std::string &request) override
    {
        if (request == "Type2")
        {
            std::cout << "Handling request of type Type2.\n";
        }
        else if (successor != nullptr)
        {
            successor->handleRequest(request);
        }
        else
        {
            std::cout << "Unable to handle the request.\n";
        }
    }
};

class ConcreteHandler3 : public Handler
{
public:
    void handleRequest(const std::string &request) override
    {
        if (request == "Type3")
        {
            std::cout << "Handling request of type Type3.\n";
        }
        else if (successor != nullptr)
        {
            successor->handleRequest(request);
        }
        else
        {
            std::cout << "Unable to handle the request.\n";
        }
    }
};

// Demo1: 日志记录器
//Logger接口
class Logger
{
public:
    virtual void log(const std::string &message) = 0;
};

//处理正常日志
class InfoLogger : public Logger
{
public:
    void log(const std::string &message) override
    {
        std::cerr << "Info: " << message << std::endl;
    }
};

//处理调试日志
class DebugLogger : public Logger
{
public:
    void log(const std::string &message) override
    {
        std::cout << "Debug: " << message << std::endl;
    }
};

//处理错误日志
class ErrorLogger : public Logger
{
public:
    void log(const std::string &message) override
    {
        std::cerr << "Error: " << message << std::endl;
    }
};

class LoggingChain
{
private:
    std::vector<std::shared_ptr<Logger>> loggers;

public:
    void addLogger(std::shared_ptr<Logger> logger)
    {
        loggers.emplace_back(logger);
    }

    void log(const std::string &message)
    {
        for (auto it = loggers.rbegin(); it != loggers.rend(); ++it)
        {
            (*it)->log(message);
        }
    }
};

// Demo2: 模拟消息接收
class Message
{
public:
    virtual ~Message() {}
};

class TextMessage : public Message
{
};

class ImageMessage : public Message
{
};

class MessageHandler
{
public:
    virtual void handle(Message *msg) = 0;
};

class TextProcessor : public MessageHandler
{
public:
    void handle(Message *msg) override
    {
        if (dynamic_cast<TextMessage *>(msg))
        {
            std::cout << "handling a text message." << std::endl;
            process(*static_cast<TextMessage *>(msg));
        }
        else
        {
            forward(msg);
        }
    }

private:
    void process(TextMessage &msg) {}

    void forward(Message *msg) {}
};

class ImageProcessor : public MessageHandler
{
public:
    void handle(Message *msg) override
    {
        if (dynamic_cast<ImageMessage *>(msg))
        {
            std::cout << "handling an image message." << std::endl;
            process(*static_cast<ImageMessage *>(msg));
        }
        else
        {
            forward(msg);
        }
    }

private:
    void process(ImageMessage &img) {}

    void forward(Message *msg) {}
};

class ChainOfResponsibility
{
public:
    void setHandler(MessageHandler *handler)
    {
        current_ = handler;
    }

    void handle(Message *msg)
    {
        current_->handle(msg);
    }

private:
    MessageHandler *current_ = nullptr;
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    Handler *handler1 = new ConcreteHandler1();
    Handler *handler2 = new ConcreteHandler2();
    Handler *handler3 = new ConcreteHandler3();

    handler1->setSuccessor(handler2);
    handler2->setSuccessor(handler3);
    handler1->handleRequest("Type2");
    handler1->handleRequest("Type3");
    handler1->handleRequest("Type4");

    delete handler1;
    delete handler2;
    delete handler3;

    std::cout << "\n--------------------------------------\n";
    LoggingChain chain;
    chain.addLogger(std::make_shared<InfoLogger>());
    chain.addLogger(std::make_shared<DebugLogger>());
    chain.addLogger(std::make_shared<ErrorLogger>());
    chain.log("This is a test message.");

    std::cout << "\n--------------------------------------\n";
    ChainOfResponsibility chain_;
    TextProcessor         txtProc;
    ImageProcessor        imgProc;
    chain_.setHandler(&txtProc);
    chain_.handle(new TextMessage());
    chain_.setHandler(&imgProc);
    chain_.handle(new ImageMessage());

    return 0;
}
