#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
    std::string userName;

    std::cout << "Please enter your name: ";
    std::cin >> userName;

    // solution logistic 1
    if (userName.size() <= 2)
        std::cout << "Your name is too short!" << std::endl;
    else
        std::cout << "Nice name, hello " << userName << "!" << std::endl;

    // solution logistic 2
    switch (userName.size())
    {
    case 0:
        std::cout << "Ah, the user with no name." 
                  << "Well, ok, hi, user with no name\n";
        break;
    
    case 1:
        std::cout << "A 1-character name? Hmm, have you read Kafka?:"
                  << "hello, " << userName << std::endl;
    
    default:
        // 长度字符串超过一个字符
        std::cout << "Hello, " << userName
                  << "-- happy to make your acquaintance!\n";
        break;
    }

    return 0;
}