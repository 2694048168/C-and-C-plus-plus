int *alloc_and_init(string file_name)
{

    ifstream infile(file_name);
    int elem_cnt;
    infile >> elem_cnt;
    int *pi = alloc_array(elem_cnt);

    int elem;
    int index = 0;
    while (infile >> elem)
    {
        pi[index++] = elem;
    }
    
    sort_array(pi, elem_cnt);
    register_data(pi);

    return pi;
}


ifstream infile(file_name); //Type do not match.

ifstream infile(file_name.c_str());
if (!infile.is_open()) //Open failed or not.
    return 0;

ifstream infile(file_name.c_str());
if (!infile.is_open())
    return 0;
int elem_cnt;
infile >> elem_cnt;
if (!infile.is_open()) //Open failed or not.
    return 0;

ifstream infile(file_name.c_str());
if (!infile.is_open())
    return 0;
int elem_cnt;
infile >> elem_cnt;
if (!infile.is_open())
    return 0;
int *pi = allocate_array(elem_cnt);
if (!pi) //Memory allocation succeed or not.
    return 0;