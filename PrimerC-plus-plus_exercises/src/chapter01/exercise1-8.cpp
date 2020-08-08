// exercise 1-8
// 练习1.8 ：指出下列哪些输出语句是合法的(如果有的话):
// std::cout << "/*;
// std::cout << "*/";
// std::cout << /* "*/" */ ;
// std::cout << /* "*/" /* "/*" */;
// 预测编译这些语句会产生什么样的结果，实际编译这些语句来验证你的答案
// ( 编写小程序，每次将上述一条语句作为其主体)，改正每个编译错误。
//

#include <iostream>

int main()
{
    // std::cout << "/*;
    /* warning: missing terminating " character
       error: missing terminating " character
       error: expected primary-expression before 'return'
    */
    
    std::cout << "*/";
    // 语法正确，输出打印 */

    std::cout << std::endl;

    //std::cout << /* "*/" */ ;
    /* warning: missing terminating " character
       error: missing terminating " character
       error: expected primary-expression before 'return'
    */

   std::cout << /* "*/" /* "/*" */;
   // 语法正确，输出打印 /*

    return 0;
}

/* 编译命令操作流程
** 0、打开终端 terminal，VSCode使用 Ctrl+shift+`; Ubuntu使用 Ctrl+Alt+T
** 1、进入当前源文件路径，cd src; cd chapter01
** 2、编译源代码文件，g++ --version; g++ -o exercise exercise1-8.cpp
** 3、运行生成的可执行程序，exercise; Ubuntu使用 ./exercise
** 4、删除生成的可执行程序，rm -rf exercise.exe; Ubuntu使用 rm -rf exercise
*/
