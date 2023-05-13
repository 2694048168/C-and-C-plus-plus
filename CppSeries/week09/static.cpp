/**
 * @file static.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-13
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the static for variable and function member of Class
 * @attention static to function of class
 *
 */

#include <iostream>
#include <cstring>

class Student
{
private:
    // static size_t student_total; /* declaration only */
    /*  since C++17, definition outside isn't needed */
    inline static size_t student_total = 0;

    char *name;
    int born;
    bool male;

public:
    Student() /* default constuctor */
    {
        ++student_total;
        name = new char[1024]{0};
        born = 0;
        male = false;
        std::cout << "Constructor: Person()" << std::endl;
    }

    Student(const char *initName, int initBorn, bool isMale)
    {
        ++student_total;
        name = new char[1024];
        setName(initName);
        born = initBorn;
        male = isMale;
        std::cout << "Constructor: Person(const char, int , bool)" << std::endl;
    }

    // default destructor
    ~Student()
    {
        --student_total;
        std::cout << "To destory object: " << name << std::endl;
        delete[] name;
    }

    void setName(const char *name)
    {
        strncpy_s(this->name, 1024, name, 1024);
    }
    void setBorn(int born)
    {
        this->born = born;
    }
    static size_t getTotal()
    {
        return student_total;
    }
    // the declarations, the definitions are out of the class
    void setGender(bool isMale);
    void printInfo() const;
};

void Student::setGender(bool isMale)
{
    male = isMale;
}

void Student::printInfo() const
{
    std::cout << "Name: " << name << std::endl;
    std::cout << "Born in " << born << std::endl;
    std::cout << "Gender: " << (male ? "Male" : "Female") << std::endl;
}

// size_t Student::student_total = 0; /* static variable definition it here */

int main(int argc, char const *argv[])
{
    std::cout << "---We have " << Student::getTotal() << " students---\n";

    Student *class1 = new Student[3]{
        {"Tom", 2000, true},
        {"Bob", 2001, true},
        {"Amy", 2002, false},
    };
    std::cout << "---We have " << Student::getTotal() << " students---\n";

    Student yu("Yu", 2000, true);
    std::cout << "---We have " << Student::getTotal() << " students---\n";

    class1[1].printInfo();
    delete []class1;
    std::cout << "---We have " << Student::getTotal() << " students---\n";

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ static.cpp
 * $ clang++ static.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */