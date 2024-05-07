/**
 * @file 13_map.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <map>
#include <stdexcept>

/**
 * @brief STL 的＜map＞头文件中可用的 std::map 是一个包含键值对的关联容器;
 * 映射(map)的键是有序且唯一的, 并且映射支持与集合相同的操作;
 * 可以将集合视为一种包含键和空值的特殊映射;因此map支持高效的插入,删除和搜索操作,并且可以使用比较器对象控制元素排序.
 * 使用映射而不是一组集合的主要优点是映射可用作关联数组, 关联数组接受键而不是整数值索引.
 * 
 * ====移动语义和复制语义: 映射的移动语义和复制语义与集合的相同;
 * ====存储模型: 映射和集合都使用红黑树作为内部结构;
 * 
 */

auto colour_of_magic     = "Colour of Magic";
auto the_light_fantastic = "The Light Fantastic";
auto equal_rites         = "Equal Rites";
auto mort                = "Mort";

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("std::map supports\n");
    // default construction
    std::map<const char *, int> emp;
    assert(emp.empty());

    // braced initialization
    std::map<const char *, int> pub_year_{
        {    colour_of_magic, 1983},
        {the_light_fantastic, 1986},
        {        equal_rites, 1987},
        {               mort, 1987},
    };
    assert(pub_year_.size() == 4);
    printf("the size of map: %lld\n", pub_year_.size());

    printf("\nstd::map 是一个具有多种访问方法的关联数组\n");
    std::map<const char *, int> pub_year{
        {    colour_of_magic, 1983},
        {the_light_fantastic, 1986},
    };
    // std::map is an associative array with
    // operator[]
    assert(pub_year[colour_of_magic] == 1983);
    pub_year[equal_rites] = 1987;
    assert(pub_year[equal_rites] == 1987);
    assert(pub_year[mort] == 0);

    // an at method
    assert(pub_year.at(colour_of_magic) == 1983);
    try
    {
        // 使用 at 仍然可以设置和检索元素,但是如果尝试访问不存在的键, 会得到一个std::out_of_range 异常
        auto elem = pub_year.at(equal_rites);
        printf("the element: %d\n", elem);
    }
    catch (std::out_of_range &exp)
    {
        printf("the std::out_of_range exception: %s\n", exp.what());
    }
    catch (const std::exception &exp)
    {
        printf("the exception: %s\n", exp.what());
    }

    printf("\nstd::map supports insert\n");
    std::map<const char *, int> pub_year1;
    pub_year1.insert({colour_of_magic, 1983});
    assert(pub_year1.size() == 1);

    std::pair<const char *, int> fp{the_light_fantastic, 1986};
    pub_year1.insert(fp);
    assert(pub_year1.size() == 2);

    auto [itr, is_new] = pub_year.insert({the_light_fantastic, 9999});
    assert(itr->first == the_light_fantastic);
    assert(itr->second == 1986);
    // 返回值is_new 表示没有插入新元素, 映射仍然只有两个元素, 此行为反映了集合的插入行为
    assert(pub_year1.size() == 2);

    printf("We can remove std::map elements using\n");
    std::map<const char *, int> pub_year2{
        {colour_of_magic, 1983},
        {           mort, 1987},
    };
    // erase
    pub_year2.erase(mort);
    assert(pub_year2.find(mort) == pub_year.end());
    // clear
    pub_year.clear();
    assert(pub_year2.empty());

    /**
     * @brief STL 的 ＜map＞ 头文件中可用的 std::multimap 是一个关联容器,
     * 这种容器包含非唯一键的键值对.
     * 无序映射和无序 multimap.
     */

    return 0;
}
