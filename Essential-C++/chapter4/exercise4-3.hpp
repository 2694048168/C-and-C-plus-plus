#ifndef EXERCISE4_3_H
#define EXERCISE4_3_H

#include <string>

using std::string;

class globalWrapper
{
public:
    static int tests_passed(){ return _tests_passed; }
    static int tests_run(){ return _tests_run; }
    static int version_number(){ return _version_number; }
    static string version_stamp(){ return _version_stamp; }
    static string program_name(){ return _program_name; }

    static void tests_passed(int nval){ _tests_passed = nval; }
    static void tests_run(int nval){ _tests_run = nval; }
    static void version_number(int nval){ _version_number = nval; }
    static void version_stamp(const string& nstamp){ _version_stamp = nstamp; }
    static void program_name(const string& npn){ _program_name = npn; }

private:
    static string _program_name;
    static string _version_stamp;
    static int _version_number;
    static int _tests_run;
    static int _tests_passed;
};

string globalWrapper::_program_name;
string globalWrapper::_version_stamp;
int globalWrapper::_version_number;
int globalWrapper::_tests_run;
int globalWrapper::_tests_passed;


#endif  // EXERCISE4_3_H 