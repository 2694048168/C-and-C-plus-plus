#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::cerr;
using std::endl;
using std::ostream;
using std::cout;


// function object
class LessThan
{
public:
    bool operator()(const string & s1, const string & s2)
    {
        return s1.size() < s2.size();
    }
};

template <typename elemType>
void display_vector(const vector<elemType> &vec, ostream &os=cout, int len=8);


int main(int argc, char**argv)
{
    ifstream ifile("./MooCat.txt");
    ofstream ofile("./MooCat_sort.txt");
    if (! ifile || ! ofile)
    {
        cerr << "Unable to open file -- bailing out!" << endl;
        return -1;
    }

    vector<string> text;
    string word;
    
    while (ifile >> word)
    {
        text.push_back(word);
    }

    sort(text.begin(), text.end(), LessThan());

    display_vector(text, ofile);

    return 0;
}

template <typename elemType>
void display_vector(const vector<elemType> &vec, ostream &os, int len)
{
    typename vector<elemType>::const_iterator iter = vec.begin(), end_it = vec.end();

    int elem_cnt = 1;
    while (iter != end_it)
    {
        os << *iter++ << (!(elem_cnt++ % len) ? '\n' : ' ');
    }
    os << endl;
    
}