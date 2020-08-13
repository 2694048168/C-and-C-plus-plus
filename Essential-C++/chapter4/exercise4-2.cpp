#include "Stack.hpp"
#include <iostream>

int main(int argc, char**argv)
{
    Stack st;
    string str;
    
    cout << "Please enter a sequence of string: ";
    while (cin >> str && ! st.full())
    {
        st.push(str);
    }
    
    if (st.empty())
    {
        cout << '\n' << "Oops: no strings were read -- bailing out\n ";
        return 0;
    }

    cout << '\n' << "Read in " << st.size() << " strings!" << endl;
    cin.clear();

    cout << "what word to search for? : ";
    cin >> str;

    bool found = st.find(str);
    int count = found ? st.count(str) : 0;

    cout << str << (found ? " is " : "isn\'t") << "in the stack.\t";
    if (found)
    {
        cout << "It occurs " << count << " times. " << endl;
    }

    return 0;
}
