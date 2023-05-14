/**
 * @file friend_class.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief friend class
 * @attention 
 *
 */

#include <iostream>

class Sniper
{
private:
    int bullets;

public:
    Sniper(int bullets = 0) : bullets(bullets) {}

    friend class Supplier;
};

class Supplier
{
private:
    int storage;

public:
    Supplier(int storage = 1000) : storage(storage) {}

    bool provide(Sniper &sniper)
    {
        // bullets is a private member
        if (sniper.bullets < 20) // no enough bullets
        {
            if (this->storage > 100)
            {
                sniper.bullets += 100;
                this->storage -= 100;
            }
            else if (this->storage > 0)
            {
                sniper.bullets += this->storage;
                this->storage = 0;
            }
            else
                return false;
        }
        std::cout << "sniper has " << sniper.bullets << " bullets now.\n";
        return true;
    }
};

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    Sniper sniper(2);
    Supplier supplier(2000);
    supplier.provide(sniper);

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ assert.cpp
 * $ clang++ assert.cpp -std=c++17
 * $ clang++ assert.cpp -DNDEBUG
 * $ clang++ assert.cpp -DNDEBUG -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */