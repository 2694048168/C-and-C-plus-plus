#include "Stack_5_1.hpp"
#include "LIFO_Stack_5_1.hpp"
#include "Peekback_Stack_5_1.hpp"

void peek(Stack &st, int index)
{
    cout << endl;
    string t;
    if (st.peek(index, t))
    {
        cout << "peek: " << t;
    }
    else
    {
        cout << "peekk failed!";
    }

    cout << endl;
}


int main(int argc, char*argv[])
{
    LIFO_Stack st;
    string str;
    while (cin >> str && ! st.full())
    {
        st.push(str);
    }

    cout << '\n' << "About to call peek() with LIFO_Stack" << endl;
    peek(st, st.top()-1);
    cout << st;

    Peekback_Stack pst;
    while (! st.empty())
    {
        string t;
        if (st.pop(t))
        {
            pst.push(t);
        }
    }

    cout << "About to call peek() with Peekbak_Stack" << endl;
    peek(pst, pst.top()-1);
    cout << pst;

    return 0;
}