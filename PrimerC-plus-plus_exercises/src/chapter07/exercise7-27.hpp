/* exercise 7-27
** 练习7.27: 给你自己的Screen类添加move、set、和display函数，
** 通过执行下面的代码检验你的类是否正确
**   Screen myScreen(5, 5, 'X');
**   myScreen.move(4, 0).set('#').display(cout);
**   cout << "\n";
**   myScreen.display(cout);
**   cout<< "\n";
**
*/

#ifndef EXERCISE7_27_H
#define EXERCISE7_27_H

#include <string>
#include <iostream>

class Screen 
{
public:
    using pos = std::string::size_type;

    Screen() = default; // 1
    Screen(pos ht, pos wd):height(ht), width(wd), contents(ht*wd, ' '){ } // 2
    Screen(pos ht, pos wd, char c):height(ht), width(wd), contents(ht*wd, c){ } // 3

    char get() const { return contents[cursor]; }
    char get(pos r, pos c) const { return contents[r*width+c]; }
    inline Screen& move(pos r, pos c);
    inline Screen& set(char c);
    inline Screen& set(pos r, pos c, char ch);

    const Screen& display(std::ostream &os) const { do_display(os); return *this; }
    Screen& display(std::ostream &os) { do_display(os); return *this; }

private:
    void do_display(std::ostream &os) const { os << contents; }

private:
    pos cursor = 0;
    pos height = 0, width = 0;
    std::string contents;
};

inline Screen& Screen::move(pos r, pos c)
{
    cursor = r*width + c;
    return *this;
}

inline Screen& Screen::set(char c)
{
    contents[cursor] = c;
    return *this;
}

inline Screen& Screen::set(pos r, pos c, char ch)
{
    contents[r*width+c] = ch;
    return *this;
}


#endif // EXERCISE7_27_H
