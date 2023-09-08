/**
 * @file 16_10_3_hangman.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cstddef>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// global variable
// const unsigned int NUM = 26;

// const std::string word_list[NUM]
//     = {"apiary", "beetle", "cereal", "danger", "ensign", "florid", "garage", "health", "insult",
//        "jackal", "keeper", "loaner", "manage", "nonce",  "onset",  "plaid",  "quilt",  "remote",
//        "stolid", "train",  "useful", "valid",  "whence", "xenon",  "yearn",  "zippy"};

/**
 * @brief 编写C++程序，读写文件和 std::string
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const char *filename = "./test.txt";

    // ========= reading word from file into vector
    std::ifstream file_reader;
    file_reader.open(filename, std::ios::in);
    if (!file_reader.is_open())
    {
        std::cout << "read file is not successfully, please check." << filename << "\n";
        return -1;
    }

    std::vector<std::string> word_list;

    std::string buffer;
    while (!file_reader.eof())
    {
        file_reader >> buffer;
        word_list.push_back(buffer);
    }

    file_reader.close();
    /* ------------------------------------- */
    const size_t NUM = word_list.size();
    std::cout << "The total number of word is: " << NUM << "\n";

    std::srand(std::time(0));

    char play;
    std::cout << "Will you play a word game? <y/n> ";
    std::cin >> play;
    play = std::tolower(play);
    while (play == 'y')
    {
        std::string target = word_list[std::rand() % NUM];

        int length = target.length();

        std::string attempt(length, '-');
        std::string bad_chars;

        unsigned int guesses = 6;
        std::cout << "Guess my secret word.\nIt has " << length << " letters, and you guess\n"
                  << "one letter at a time.\nYou get " << guesses << " wrong guesses.\n";

        std::cout << "Your word: " << attempt << "\n";
        while (guesses > 0 && attempt != target)
        {
            char letter;
            std::cout << "Guess one letter: ";
            std::cin >> letter;

            if (bad_chars.find(letter) != std::string::npos || attempt.find(letter) != std::string::npos)
            {
                std::cout << "You already guessed that. Try again.\n";
                continue;
            }

            int loc = target.find(letter);
            if (loc == std::string::npos)
            {
                std::cout << "Oh, bad guess!\n";
                --guesses;

                bad_chars += letter; // add to string
            }
            else
            {
                std::cout << "Good guess!\n";
                attempt[loc] = letter;

                // check if letter appears again
                loc = target.find(letter, loc + 1);
                while (loc != std::string::npos)
                {
                    attempt[loc] = letter;

                    loc = target.find(letter, loc + 1);
                }
            }
            std::cout << "Your word: " << attempt << "\n";
            if (attempt != target)
            {
                if (bad_chars.length() > 0)
                    std::cout << "Bad choices: " << bad_chars << "\n";

                std::cout << guesses << " bad guesses left\n";
            }
        }
        if (guesses > 0)
            std::cout << "That's right!\n";
        else
            std::cout << "Sorry, the word is " << target << ".\n";

        std::cout << "Will you play another? <y/n> ";
        std::cin >> play;
        play = std::tolower(play);
    }

    std::cout << "Bye\n";

    return 0;
}
