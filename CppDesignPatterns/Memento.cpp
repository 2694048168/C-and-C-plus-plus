/**
 * @file Memento.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-03
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Memento.cpp -std=c++20
 * clang++ Memento.cpp -std=c++20
 * 
 */

#include <iostream>
#include <string>
#include <vector>

// Demo1
class Originator
{
private:
    std::string state;

public:
    void SetState(const std::string &newState)
    {
        state = newState;
    }

    std::string GetState() const
    {
        return state;
    }

    class Memento
    {
    private:
        std::string state;

    public:
        Memento(const std::string &originatorState)
        {
            state = originatorState;
        }

        std::string GetSavedState() const
        {
            return state;
        }
    };

    Memento CreateMemento() const
    {
        return Memento(state);
    }

    void RestoreState(const Memento &memento)
    {
        state = memento.GetSavedState();
    }
};

class Caretaker
{
private:
    std::vector<Originator::Memento> mementos;

public:
    void AddMemento(const Originator::Memento &memento)
    {
        mementos.push_back(memento);
    }

    Originator::Memento GetMemento(int index) const
    {
        if (index >= 0 && index < mementos.size())
        {
            return mementos[index];
        }
        throw std::out_of_range("Invalid Memento index");
    }
};

// Demo2
class Memento
{
private:
    int state;

public:
    Memento(int s)
        : state(s)
    {
    }

    int getState() const
    {
        return state;
    }
};

class Originator_
{
private:
    int currentState;

public:
    void setState(int newState)
    {
        currentState = newState;
    }

    Memento createMemento()
    {
        return Memento(currentState);
    }

    void restoreFromMemento(Memento &memento)
    {
        currentState = memento.getState();
    }

    void getState()
    {
        std::cout << "Current state: " << currentState << std::endl;
    }
};

class CareTaker_
{
private:
    std::vector<Memento> memoranda;

public:
    void saveMemento(Originator_ &originator)
    {
        memoranda.push_back(originator.createMemento());
    }

    void restoreOriginatorToState(Originator_ &originator, size_t index)
    {
        originator.restoreFromMemento(memoranda[index]);
    }

    Memento getMemento(int index) const
    {
        if (index >= 0 && index < memoranda.size())
        {
            return memoranda[index];
        }
        throw std::out_of_range("Invalid Memento index");
    }
};

// Demo: 模拟文本编辑器的撤销功能
class TextMemento
{
public:
    TextMemento(const std::string &text)
    {
        text_ = text;
    }

    const std::string &getText() const
    {
        return text_;
    }

private:
    std::string text_;
};

class TextEditor
{
public:
    void setText(const std::string &text)
    {
        text_ = text;
    }

    const std::string &getText() const
    {
        return text_;
    }

    TextMemento createMemento()
    {
        return TextMemento(text_);
    }

    void restoreMemento(const TextMemento &memento)
    {
        text_ = memento.getText();
    }

private:
    std::string text_;
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    Originator originator;
    Caretaker  caretaker;

    originator.SetState("State 1");
    caretaker.AddMemento(originator.CreateMemento());
    originator.SetState("State 2");
    caretaker.AddMemento(originator.CreateMemento());
    originator.SetState("State 3");
    caretaker.AddMemento(originator.CreateMemento());

    std::cout << "Current state: " << originator.GetState() << std::endl;
    originator.RestoreState(caretaker.GetMemento(2));
    std::cout << "Current state: " << originator.GetState() << std::endl;

    std::cout << "--------------------------------------\n";
    Originator_ originator_;
    CareTaker_  caretaker_;

    originator_.setState(5);
    caretaker_.saveMemento(originator_);
    originator_.getState();

    originator_.setState(10);
    caretaker_.saveMemento(originator_);
    originator_.getState();

    originator_.setState(15);
    caretaker_.saveMemento(originator_);
    originator_.getState();

    originator_.setState(20);
    originator_.getState();

    caretaker_.restoreOriginatorToState(originator_, 1);
    originator_.getState();

    std::cout << "--------------------------------------\n";
    TextEditor editor;

    std::vector<TextMemento> history;

    editor.setText("Hello, World!");
    history.push_back(editor.createMemento());

    editor.setText("Goodbye!");
    history.push_back(editor.createMemento());
    std::cout << "Current Text: " << editor.getText() << std::endl;

    editor.restoreMemento(history[0]);
    std::cout << "After Undo: " << editor.getText() << std::endl;

    editor.restoreMemento(history[1]);
    std::cout << "After Redo: " << editor.getText() << std::endl;

    return 0;
}
