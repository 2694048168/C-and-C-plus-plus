/**
 * @file 17_json_test.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** Qt中Json的操作, Qt提供的Json类进行数据的组织和解析, 相关的类常用的主要有四个:
 * ?1. QJsonDocument:封装了一个完整的JSON文档,并且可以从UTF-8编码的基于文本的表示以及Qt自己的二进制格式读取和写入该文档;
 * ?2. QJsonArray:JSON数组是一个值列表,可以通过从数组中插入和删除QJsonValue来操作该列表;
 * ?3. QJsonObject:JSON对象是键值对的列表,其中键是唯一的字符串,值由QJsonValue表示;
 * ?4. QJsonValue:该类封装了JSON支持的数据类型;
 * 
 * QJsonValue可以封装的基础数据类型有六种(和Json支持的类型一致):
// =============================================
1. 布尔类型：QJsonValue::Bool
2. 浮点类型（包括整形）： QJsonValue::Double
3. 字符串类型： QJsonValue::String
4. Json数组类型： QJsonValue::Array
5. Json对象类型：QJsonValue::Object
6. 空值类型： QJsonValue::Null
// =============================================
 * 
 */

#include <QApplication>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <iostream>

void writeJson()
{
    QJsonObject obj;
    obj.insert("Name", "Ace");
    obj.insert("Sex", "man");
    obj.insert("Age", 20);

    QJsonObject subObj;
    subObj.insert("Father", "Gol·D·Roger");
    subObj.insert("Mother", "Portgas·D·Rouge");
    QJsonArray array;
    array.append("Sabo");
    array.append("Monkey D. Luffy");
    subObj.insert("Brother", array);
    obj.insert("Family", subObj);
    obj.insert("IsAlive", false);
    obj.insert("Comment", "yyds");

    QJsonDocument doc(obj);
    QByteArray    json = doc.toJson();

    QFile file("d:\\ace.json");
    file.open(QFile::WriteOnly);
    file.write(json);
    file.close();
}

void readJson()
{
    QFile file("d:\\ace.json");
    file.open(QFile::ReadOnly);
    QByteArray json = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(json);
    if (doc.isObject())
    {
        QJsonObject obj  = doc.object();
        QStringList keys = obj.keys();
        for (int i = 0; i < keys.size(); ++i)
        {
            QString    key   = keys.at(i);
            QJsonValue value = obj.value(key);
            if (value.isBool())
            {
                qDebug() << key << ":" << value.toBool();
            }
            if (value.isString())
            {
                qDebug() << key << ":" << value.toString();
            }
            if (value.isDouble())
            {
                qDebug() << key << ":" << value.toInt();
            }
            if (value.isObject())
            {
                qDebug() << key << ":";
                // 直接处理内部键值对, 不再进行类型判断的演示
                QJsonObject subObj = value.toObject();
                QStringList ls     = subObj.keys();
                for (int i = 0; i < ls.size(); ++i)
                {
                    QJsonValue subVal = subObj.value(ls.at(i));
                    if (subVal.isString())
                    {
                        qDebug() << "   " << ls.at(i) << ":" << subVal.toString();
                    }
                    if (subVal.isArray())
                    {
                        QJsonArray array = subVal.toArray();
                        qDebug() << "   " << ls.at(i) << ":";
                        for (int j = 0; j < array.size(); ++j)
                        {
                            // 因为知道数组内部全部为字符串, 不再对元素类型进行判断
                            qDebug() << "       " << array[j].toString();
                        }
                    }
                }
            }
        }
    }
}

// ====================================
int main(int argc, char **argv)
{
    // 创建应用程序对象, 在一个Qt项目中实例对象有且仅有一个
    // 类的作用: 检测触发的事件, 进行事件循环并处理
    QApplication app(argc, argv);

    std::cout << "==========================\n";

    // 应用程序对象开始事件循环, 保证应用程序不退出
    return app.exec();
}
