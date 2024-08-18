/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-23
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "config_params/Person.pb.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

void printInfo(ModuleName::Person pp)
{
    std::cout << pp.id() << ", " << pp.sex() << ", " << pp.age() << std::endl;
    std::cout << pp.addr().addr() << ", " << pp.addr().num() << std::endl;
    int size = pp.name_size();
    for (int i = 0; i < size; ++i)
    {
        std::cout << pp.name(i) << std::endl;
    }
    std::cout << pp.color() << std::endl;
}

// -------------------------------------
int main(int argc, const char **argv)
{
    // 序列化
    ModuleName::Person p;
    p.set_id(10);
    p.set_age(32);
    p.set_sex("man");

    p.add_name();
    p.set_name(0, "路飞");
    p.add_name("艾斯");
    p.add_name("萨博");
    p.mutable_addr()->set_addr("北京市长安区天安门");
    p.mutable_addr()->set_num(1001);
    p.set_color(ModuleName::Color::Blue);

    // 序列化对象 p, 最终得到一个字符串
    std::string output;
    p.SerializeToString(&output);

    // 反序列化数据
    ModuleName::Person pp;
    pp.ParseFromString(output);
    printInfo(pp);

    // =================================================
    // 序列化到本地磁盘文件
    // 将数据序列化写入到磁盘文件中, c++ 风格
    // ostream 子类 ofstream -> 写文件
    std::filesystem::path filename{"./conf/Person"};
    if (!std::filesystem::exists(filename.parent_path()))
        std::filesystem::create_directories(filename.parent_path());

    std::ofstream file_writer;
    file_writer.open(filename, std::ios::out | std::ios::trunc | std::ios::binary);
    // bool SerializeToOstream(std::ostream* output) const;
    p.SerializeToOstream((std::ostream *)&file_writer);
    file_writer.close();

    std::cout << "\n======= From disk conf file ============\n";
    std::ifstream file_reader;
    file_reader.open(filename, std::ios::in | std::ios::binary);
    if (!file_reader.is_open())
    {
        std::cout << "read file is not successfully, please check." << filename << "\n";
        return 0;
    }
    ModuleName::Person p_file;
    // bool ParseFromIstream(std::istream* input);
    p_file.ParseFromIstream((std::istream *)&file_reader);
    file_reader.close();
    printInfo(p_file);

    return 0;
}
