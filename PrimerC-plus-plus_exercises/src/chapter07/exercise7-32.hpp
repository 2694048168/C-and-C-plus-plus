/* exercise 7-32
** 练习7.32: 定义你自己的Screen 和 Window_mgr，其中clear是Window_mgr的成员，是Screen的友元
**
*/

#ifndef EXERCISE7_32_H
#define EXERCISE7_32_H

#include <vector>
#include <string>
#include <iostream>

class Screen;

class Window_mgr 
{
public:
    using ScreenIndex = std::vector<Screen>::size_type;
    inline void clear(ScreenIndex);
private:
    std::vector<Screen> screens;
};

class Screen 
{
    friend void Window_mgr::clear(ScreenIndex);
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

inline void Window_mgr::clear(ScreenIndex i)
{ 
    if (i >= screens.size()) return;    // judge for out_of_range.
    Screen &s = screens[i];
    s.contents = std::string(s.height * s.width, ' ');
}

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


#endif // EXERCISE7_32_H
