#include "Stack_5_2.hpp"
#include "Peekback_Stack_5_2.hpp"

void peek(Stack &st, int index)
{
    cout << endl;
    string str;
    if (st.peek(index, str))
    {
        cout << "peek: " << str;
    }
    else
    {
        cout << "peekk failed!";
    }

    cout << endl;
}


int main(int argc, char *argv[])
{
    Stack st;
    string str;
    while (cin >> str && ! st.full())
    {
        st.push(str);
    }

    cout << '\n' << "About to call peek() with Stack" << endl;
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