#include <map>
#include <set>
#include <string>
#include <iostream>
#include <fstream>

/*
** 命名空间 使用方法
** 0、在引用命名空间成员时，要用命名空间名和作用域分辨符对命名空间成员进行限定，
**    以区别不同的命名空间中的同名标识符。
**    即：命名空间名::命名空间成员名
**     std::cout
**
** 1、使用命名空间别名
**    可以为命名空间起一个别名，用来替代较长的命名空间名，
**    可以用一个较短的别名替代它。如：OpenCV 中的类型名
**    namespace s = std
**
** 2 、使用using命名空间成员名
**     using后面的命名空间成员名必须是由命名空间限定的名字
**     using std::cout
**
** 3、使用using namespace命名空间名
**    using namespace std;
**    声明了在本作用域中要用到命名空间std中的成员，在使用该命名空间的任何成员时都不必再使用命名空间限定。
*/
using std::set;
using std::string;
using std::map;
using std::ifstream;
using std::ofstream;
using std::cerr;
using std::endl;
using std::cout;
using std::cin;


void initialize_exclusion_set(set<string>&);
void process_file(map<string, int>&, const set<string>&, ifstream&);
void user_query(const map<string, int>&);
void dispaly_word_count(const map<string, int>&, ofstream&);


int main(int argc, char**argv)
{
    ifstream ifile("./column.txt");
    ofstream ofile("./column_map.txt");
    if (! ifile || ! ofile)
    {
        cerr << "Unable to open file -- bailing out!" << endl;
        return -1;
    }

    set<string> exclude_set;
    initialize_exclusion_set(exclude_set);

    map<string, int> word_count;
    process_file(word_count, exclude_set, ifile);

    user_query(word_count);

    dispaly_word_count(word_count, ofile);

    return 0;
}


void initialize_exclusion_set(set<string> &exs)
{
    static string _excluded_words[25] = {
        "the", "and", "but", "that", "then", "are", "been",
        "can", "a", "could", "did", "for", "of",
        "had", "have", "him", "his", "her", "its", "is",
        "were", "which", "when", "with","would"
    };

    exs.insert(_excluded_words, _excluded_words+25);
}


void process_file(map<string, int> &word_count, const set<string> &exclude_set, ifstream &ifile)
{
    string word;
    while (ifile >> word)
    {
        if (exclude_set.count(word))
        {
            continue;
        }

        word_count[word]++;
    }
}


void user_query(const map<string, int> &word_map)
{
    string search_word;
    cout << "Please enter a word to search(q to quit): ";
    cin >> search_word;

    while (search_word.size() && search_word != "q")
    {
        map<string, int>::const_iterator it;
        if ((it = word_map.find(search_word)) != word_map.end())
        {
            cout << "Found! " << it->first
                 << " occurs " << it->second
                 << " times." << endl;
        }
        else
        {
            cout << search_word << " was not found in text." << endl;
        }

        cout << "\nAnother search? (q to quit) ";
        cin >> search_word;
    }
}


void dispaly_word_count(const map<string, int> &word_map, ofstream &os)
{
    map<string, int>::const_iterator iter = word_map.begin(), end_it = word_map.end();

    while (iter != end_it)
    {
        os << iter->first << "(" << iter->second << ")" << endl;
        ++iter;
    }
    os << endl;
}