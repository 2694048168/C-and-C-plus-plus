/**
 * @file Proxy.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-24
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Proxy.cpp -std=c++20
 * clang++ Proxy.cpp -std=c++20
 * 
 */

#include <iostream>
#include <string>

class Subject
{
public:
    virtual void request() = 0;

    virtual ~Subject() {}
};

class RealSubject : public Subject
{
public:
    void request() override
    {
        std::cout << "RealSubject.request()\n";
    }
};

class Proxy : public Subject
{
private:
    Subject *realSubject;

public:
    Proxy()
        : realSubject{new RealSubject()}
    {
    }

    ~Proxy()
    {
        if (realSubject)
        {
            delete realSubject;
            realSubject = nullptr;
        }
    }

    // Forward calls to the RealSubject:
    void request() override
    {
        if (!realSubject)
            return;

        realSubject->request();
    }
};

// 开发场景: 基于Proxy模式来模拟对图片的延迟加载和显示控制
class Image
{
public:
    virtual void display() = 0;
    virtual ~Image()       = default;
};

class RealImage : public Image
{
private:
    std::string filename;

public:
    RealImage(const std::string &filename)
        : filename(filename)
    {
        // Simulate loading the image
        std::cout << "Loading image: " << filename << std::endl;
    }

    void display() override
    {
        std::cout << "Displaying image: " << filename << std::endl;
    }
};

class ImageProxy : public Image
{
private:
    // Reference to the Real Object
    RealImage  *realImage;
    std::string filename;

public:
    ImageProxy(const std::string &filename)
        : filename(filename)
        , realImage(nullptr)
    {
    }

    void display() override
    {
        if (realImage == nullptr)
        {
            realImage = new RealImage(filename);
        }
        realImage->display();
    }
};

// -----------------------------------
int main(int argc, const char **argv)
{
    Proxy proxy;
    proxy.request();

    std::cout << "-----------------------------\n";
    //Create a proxy to an image
    Image *image = new ImageProxy("images/Proxy.png");

    //Display the image
    image->display();

    //Displaying the image again
    image->display();

    if (image)
    {
        delete image;
        image = nullptr;
    }

    return 0;
}
