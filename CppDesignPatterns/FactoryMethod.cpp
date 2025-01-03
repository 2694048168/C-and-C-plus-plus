/**
 * @file FactoryMethod.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-03
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ FactoryMethod.cpp -std=c++20
 * clang++ FactoryMethod.cpp -std=c++20
 * 
 */

#include <iostream>
#include <list>
#include <string>

class Product
{
public:
    virtual void use() = 0;

    virtual ~Product() = default;
};

class ConcreteProductA : public Product
{
public:
    void use()
    {
        std::cout << "Using ConcreteProductA" << std::endl;
    }
};

class ConcreteProductB : public Product
{
public:
    void use()
    {
        std::cout << "Using ConcreteProductB" << std::endl;
    }
};

class Factory
{
public:
    virtual Product *createProduct() = 0;

    virtual ~Factory() = default;
};

class ConcreteFactoryA : public Factory
{
public:
    Product *createProduct()
    {
        return new ConcreteProductA();
    }
};

class ConcreteFactoryB : public Factory
{
public:
    Product *createProduct()
    {
        return new ConcreteProductB();
    }
};

// ---------------------
//Abstract Product
class Page
{
public:
    virtual std::string GetPageName(void) = 0;
};

//Concrete Product
class SkillsPage : public Page
{
public:
    std::string GetPageName(void)
    {
        return "SkillsPage";
    }
};

//Concrete Product
class EducationPage : public Page
{
public:
    std::string GetPageName(void)
    {
        return "EducationPage";
    }
};

//Concrete Product
class ExperiencePage : public Page
{
public:
    std::string GetPageName(void)
    {
        return "ExperiencePage";
    }
};

//Concrete Product
class IntroductionPage : public Page
{
public:
    std::string GetPageName(void)
    {
        return "IntroductionPage";
    }
};

//Concrete Product
class ResultsPage : public Page
{
public:
    std::string GetPageName(void)
    {
        return "ResultsPage";
    }
};

//Concrete Product
class ConclusionPage : public Page
{
public:
    std::string GetPageName(void)
    {
        return "ConclusionPage";
    }
};

//Concrete Product
class SummaryPage : public Page
{
public:
    std::string GetPageName(void)
    {
        return "SummaryPage";
    }
};

//Concrete Product
class BibliographyPage : public Page
{
public:
    std::string GetPageName(void)
    {
        return "BibliographyPage";
    }
};

//Abstract Factory
class Document
{
public:
    Document() {}

    void AddPages(Page *page)
    {
        pages_.push_back(page);
    }

    const std::list<Page *> &GetPages(void)
    {
        return pages_;
    }

    //Factory Method
    virtual void CreatePages(void) = 0;

private:
    std::list<Page *> pages_;
};

//Concrete Factory
class Resume : public Document
{
public:
    Resume()
    {
        CreatePages();
    }

    void CreatePages(void)
    {
        AddPages(new SkillsPage());
        AddPages(new EducationPage());
        AddPages(new ExperiencePage());
    }
};

//Concrete Factory
class Report : public Document
{
public:
    Report()
    {
        CreatePages();
    }

    void CreatePages(void)
    {
        AddPages(new SummaryPage());
        AddPages(new IntroductionPage());
        AddPages(new ResultsPage());
        AddPages(new ConclusionPage());
        AddPages(new BibliographyPage());
    }
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    Factory *factoryA = new ConcreteFactoryA();
    Product *productA = factoryA->createProduct();
    productA->use();

    Factory *factoryB = new ConcreteFactoryB();
    Product *productB = factoryB->createProduct();
    productB->use();

    delete factoryA;
    delete productA;
    delete factoryB;
    delete productB;

    std::cout << "--------------------------------------\n";
    Document          *doc1 = new Resume();
    Document          *doc2 = new Report();
    //Get and print the pages of the first document
    std::list<Page *> &doc1Pages = const_cast<std::list<Page *> &>(doc1->GetPages());
    std::cout << "\nResume Pages -------------" << std::endl;
    for (std::list<Page *>::iterator it = doc1Pages.begin(); it != doc1Pages.end(); it++)
    {
        std::cout << "\t" << (*it)->GetPageName() << std::endl;
    }
    //Get and print the pages of the second document
    std::list<Page *> &doc2Pages = const_cast<std::list<Page *> &>(doc2->GetPages());
    std::cout << "\nReport Pages -------------" << std::endl;
    for (std::list<Page *>::iterator it = doc2Pages.begin(); it != doc2Pages.end(); it++)
    {
        std::cout << "\t" << (*it)->GetPageName() << std::endl;
    }

    return 0;
}
