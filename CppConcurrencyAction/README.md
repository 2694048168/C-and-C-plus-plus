## C++ 并发编程实战

> Talk is cheap, show me the code.

### features
0. Concurrency and Thread
    - [x] CSP Concurrency Model
    - [x] thread-safe queue
    - [x] thread pool and thread & task
1. Hello, world of concurrency in C++, This chapter covers
    - [x] What is meant by concurrency and multithreading
    - [x] Why you might want to use concurrency and multithreading in your applications
    - [x] Some of the history of the support for concurrency in C++
    - [x] What a simple multithreaded C++ program looks like
    - [x] e.g. <font color=#e464f5>01_hello.cpp</font>
2. Managing threads, This chapter covers
    - [x] Starting threads, and various ways of specifying code to run on a new thread
    - [x] Waiting for a thread to finish versus leaving it to run
    - [x] Uniquely identifying threads
    - [x] e.g. <font color=#e464f5>02_start_thread.cpp</font> and <font color=#e464f5>03_thread_parameter.cpp</font>
3. Sharing data between threads, This chapter covers
    - [ ] Problems with sharing data between threads
    - [ ] Protecting data with mutexes
    - [ ] Alternative facilities for protecting shared data
    - [ ] e.g. 
4. Synchronizing concurrent operations, This chapter covers
    - [ ] Waiting for an event
    - [ ] Waiting for one-off events with futures
    - [ ] Waiting with a time limit
    - [ ] Using the synchronization of operations to simplify code
    - [ ] e.g.
5. The C++ memory model and operations on atomic types, This chapter covers
    - [ ] The details of the C++ memory model
    - [ ] The atomic types provided by the C++
    - [ ] Standard Library
    - [ ] The operations that are available on those types
    - [ ] How those operations can be used to provide synchronization between threads
    - [ ] e.g.
6. Designing lock-based concurrent data structures, This chapter covers
    - [ ] What it means to design data structures for concurrency
    - [ ] Guidelines for doing so
    - [ ] Example implementations of data structures designed for concurrency
    - [ ] e.g.
7. Designing lock-free concurrent data structures, This chapter covers
    - [ ] Implementations of data structures designed for concurrency without using locks
    - [ ] Techniques for managing memory in lock-free data structures
    - [ ] Simple guidelines to aid in the writing of lock-free data structures
    - [ ] e.g.
8. Designing concurrent code, This chapter covers
    - [ ] Techniques for dividing data between threads
    - [ ] Factors that affect the performance of concurrent code
    - [ ] How performance factors affect the design of data structures
    - [ ] Exception safety in multithreaded code
    - [ ] Scalability
    - [ ] Example implementations of several parallel algorithms
    - [ ] e.g.
9. Advanced thread management, This chapter covers
    - [ ] Thread pools
    - [ ] Handling dependencies between pool tasks
    - [ ] Work stealing for pool threads
    - [ ] Interrupting threads
    - [ ] e.g.
10. Parallel algorithms, This chapter covers
    - [ ] Using the C++17 parallel algorithms
    - [ ] e.g.
11. Testing and debugging multithreaded applications, This chapter covers
    - [ ] Concurrency-related bugs
    - [ ] Locating bugs through testing and code review
    - [ ] Designing multithreaded tests
    - [ ] Testing the performance of multithreaded code
    - [ ] e.g.
12. appendix A, Brief reference for some C++11 modern language features
13. appendix B, Brief comparison of concurrency libraries
14. appendix C, A message-passing framework and complete ATM example
15. appendix D, C++ Thread Library reference


### quick start
```shell
# clone the source code into a folder
git clone --recursive https://github.com/2694048168/C-and-C-plus-plus.git
cd C-and-C-plus-plus/CppConcurrencyAction

# compile the C++ source code
clang++ 01_hello.cpp -std=c++17 -o hello
g++ 01_hello.cpp -std=c++17 -o hello

# executable the binary file
./hello
```


### reference
- [C++ Concurrency In Action SECOND EDITION 中文翻译版本](https://github.com/xiaoweiChen/CPP-Concurrency-In-Action-2ed-2019)
