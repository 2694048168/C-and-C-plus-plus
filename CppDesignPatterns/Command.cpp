/**
 * @file Command.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-02
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Command.cpp -std=c++20
 * clang++ Command.cpp -std=c++20
 * 
 */

#include <iostream>
#include <string>
#include <vector>

class Command
{
public:
    virtual void execute() = 0;
    virtual ~Command()     = default;
};

class ConcreteCommand : public Command
{
private:
    std::string receiver_;

public:
    ConcreteCommand(const std::string &receiver)
    {
        receiver_ = receiver;
    }

    void execute() override
    {
        std::cout << "ConcreteCommand: " << receiver_ << "\n";
    }
};

class Invoker
{
private:
    std::vector<Command *> commands_;

public:
    void addCommand(Command *command)
    {
        commands_.push_back(command);
    }

    void executeCommands()
    {
        for (auto command : commands_)
        {
            command->execute();
        }
        commands_.clear();
    }
};

class Receiver
{
public:
    Receiver(std::string cmd_str)
    {
        cmd = cmd_str;
    }

    void action()
    {
        std::cout << "Operating " << cmd << std::endl;
    }

private:
    std::string cmd;
};

class ConcreteCommand_ : public Command
{
private:
    Receiver *receiver;

public:
    ConcreteCommand_(Receiver *receiver)
    {
        this->receiver = receiver;
    }

    void execute() override
    {
        receiver->action();
    }
};

class Invoker_
{
private:
    std::vector<Command *> commands;

public:
    void addCommand(Command *command)
    {
        commands.push_back(command);
    }

    void executeCommands()
    {
        for (auto command : commands)
        {
            command->execute();
        }
        commands.clear();
    }
};

// 基于命令模式实现的模拟远程灯光控制
//Receiver
class Light
{
public:
    void on()
    {
        std::cout << "The light is on\n";
    }

    void off()
    {
        std::cout << "The light is off\n";
    }
};

class LightOnCmd : public Command
{
public:
    LightOnCmd(Light *light)
    {
        mLight = light;
    }

    void execute()
    {
        mLight->on();
    }

private:
    Light *mLight;
};

class LightOffCmd : public Command
{
public:
    LightOffCmd(Light *light)
    {
        mLight = light;
    }

    void execute()
    {
        mLight->off();
    }

private:
    Light *mLight;
};

//Invoker
class RemoteControl
{
public:
    void setCommand(Command *cmd)
    {
        mCmd = cmd;
    }

    void buttonPressed()
    {
        mCmd->execute();
    }

private:
    Command *mCmd;
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    Invoker  invoker;
    Command *command1 = new ConcreteCommand("command 01 -> ");
    Command *command2 = new ConcreteCommand("command 02 -> ");
    invoker.addCommand(command1);
    invoker.addCommand(command2);
    invoker.executeCommands();
    delete command1;
    delete command2;

    std::cout << "--------------------------------------\n";
    Receiver *receiver1 = new Receiver("action_01");
    Receiver *receiver2 = new Receiver("action_02");
    Command  *command1_ = new ConcreteCommand_(receiver1);
    Command  *command2_ = new ConcreteCommand_(receiver2);

    Invoker_ invoker_;
    invoker_.addCommand(command1_);
    invoker_.addCommand(command2_);
    invoker_.executeCommands();

    delete command1_;
    delete command2_;
    delete receiver1;
    delete receiver2;

    std::cout << "--------------------------------------\n";
    Light       *light    = new Light;
    LightOnCmd  *lightOn  = new LightOnCmd(light);
    LightOffCmd *lightOff = new LightOffCmd(light);

    RemoteControl *control = new RemoteControl;

    control->setCommand(lightOn);
    control->buttonPressed();
    control->setCommand(lightOff);
    control->buttonPressed();

    delete light;
    delete lightOn;
    delete lightOff;
    delete control;

    return 0;
}
