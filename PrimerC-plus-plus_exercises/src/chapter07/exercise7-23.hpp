/* exercise 7-23
** 练习7.23: 编写你自己的Screen类。
** solution: 
**
*/

#ifndef EXERCISE7_23_H
#define EXERCISE7_23_H

#include <string>

class Screen 
{
    public:
        using pos = std::string::size_type;

        Screen() = default;
        Screen(pos ht, pos wd, char c):height(ht), width(wd), contents(ht*wd, c){ }

        char get() const { return contents[cursor]; }
        char get(pos r, pos c) const { return contents[r*width+c]; }

    private:
        pos cursor = 0;
        pos height = 0, width = 0;
        std::string contents;
};

#endif // EXERCISE7_23_H
