/* exercise 7-53、7-54、7-55
** 练习7.53: 定义你自己的Debug。
** solution:
**
** 练习7.54: Debug 中以set_开头的成员应该被声明成constexpr吗？如果不，为什么？
** solution：不能。constexpr函数有且只能包含一条return语句。
**
** 练习7.55: 7.5.5节（第266页）的 Data 类是字面值常量类吗？请解释原因。
** solution: 不是。s的数据类型string不是字面值类型。数据类型都是字面值类型的聚合类是字面值常量类
** 
*/

#ifndef EXERCISE7_53_54_55_H
#define EXERCISE7_53_54_55_H

// solution 7-53
class Debug 
{
public:
    constexpr Debug(bool b = true) : rt(b), io(b), other(b) { }
    constexpr Debug(bool r, bool i, bool o) : rt(r), io(i), other(0) { }
    constexpr bool any() { return rt || io || other; }
    
    void set_rt(bool b) { rt = b; }
    void set_io(bool b) { io = b; }
    void set_other(bool b) { other = b; }
    
private:
    bool rt;        // runtime error
    bool io;        // I/O error
    bool other;     // the others
};


#endif // EXERCISE7_53_54_55_H
