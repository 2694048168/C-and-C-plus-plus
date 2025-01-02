/**
 * @file State.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-02
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ State.cpp -std=c++20
 * clang++ State.cpp -std=c++20
 * 
 */

#include <iostream>
#include <thread>

class State
{
public:
    virtual ~State() {}

    virtual void handle() = 0;
};

class ConcreteStateA : public State
{
public:
    void handle() override
    {
        std::cout << "Handling state A" << std::endl;
    }
};

class ConcreteStateB : public State
{
public:
    void handle() override
    {
        std::cout << "Handling state B" << std::endl;
    }
};

class Context
{
private:
    State *state;

public:
    Context(State *state)
        : state(state)
    {
    }

    ~Context()
    {
        delete state;
    }

    void setState(State *state)
    {
        delete this->state;
        this->state = state;
    }

    void request()
    {
        state->handle();
    }
};

// Demo1: 基于状态模式模拟的交通信号灯
class TrafficLightState
{
public:
    virtual void handleState() = 0;
};

class GreenState : public TrafficLightState
{
public:
    void handleState() override
    {
        std::cout << "Traffic Light: Green!" << std::endl;
    }
};

class RedState : public TrafficLightState
{
public:
    void handleState() override
    {
        std::cout << "Traffic Light: Red!" << std::endl;
    }
};

class YellowState : public TrafficLightState
{
public:
    void handleState() override
    {
        std::cout << "Traffic Light: Yellow!" << std::endl;
    }
};

class TrafficLight
{
private:
    TrafficLightState *currentState;

public:
    TrafficLight(TrafficLightState *initialState)
    {
        currentState = initialState;
    }

    void changeState(TrafficLightState *newState)
    {
        currentState = newState;
    }

    void operate()
    {
        currentState->handleState();
    }
};

// Demo2: 基于状态模式模拟的网络管理
class NetState;

class TCPConnection
{
public:
    TCPConnection();
    void open();
    void close();
    void setState(NetState *newState);

private:
    NetState *currentState;
};

class NetState
{
public:
    virtual void open(TCPConnection *connection)  = 0;
    virtual void close(TCPConnection *connection) = 0;
};

class ReOpenState : public NetState
{
public:
    void open(TCPConnection *connection) override
    {
        std::cout << "[State3]Network is already ReOpen" << std::endl;
    }

    void close(TCPConnection *connection) override
    {
        std::cout << "[State3]Closing Network" << std::endl;
    }
};

class ClosedState : public NetState
{
public:
    void open(TCPConnection *connection) override
    {
        std::cout << "[State2]Opening Network" << std::endl;
        connection->setState(new ReOpenState());
    }

    void close(TCPConnection *connection) override
    {
        std::cout << "[State2]Network is already closed" << std::endl;
    }
};

class OpenState : public NetState
{
public:
    void open(TCPConnection *connection) override
    {
        std::cout << "[State1]Network is already open" << std::endl;
    }

    void close(TCPConnection *connection) override
    {
        std::cout << "[State1]Closing Network" << std::endl;
        connection->setState(new ClosedState());
    }
};

TCPConnection::TCPConnection()
    : currentState(new OpenState())
{
}

void TCPConnection::open()
{
    currentState->open(this);
}

void TCPConnection::close()
{
    currentState->close(this);
}

void TCPConnection::setState(NetState *newState)
{
    currentState = newState;
}

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    State *stateA = new ConcreteStateA();
    State *stateB = new ConcreteStateB();

    Context context(stateA);
    context.request();

    context.setState(stateB);
    context.request();

    std::cout << "--------------------------------------\n";
    GreenState  greenState;
    RedState    redState;
    YellowState yellowState;

    TrafficLight trafficLight(&greenState);
    trafficLight.operate();

    trafficLight.changeState(&yellowState);
    trafficLight.operate();
    trafficLight.changeState(&redState);
    trafficLight.operate();

    std::cout << "--------------------------------------\n";
    TCPConnection tcpConnection;

    tcpConnection.open();
    tcpConnection.close();

    tcpConnection.open();
    tcpConnection.close();

    return 0;
}
