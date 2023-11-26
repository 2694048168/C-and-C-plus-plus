#ifndef __DATA_STRUCT_HPP__
#define __DATA_STRUCT_HPP__

#include <QJsonObject>


struct dataStruct
{
    const char* name;
    int age;
    bool gender;
    const char* company;
    double weight;

    QJsonObject toJson() const
    {
        QJsonObject obj;

        obj["name"] = name;
        obj["age"] = age;
        obj["gender"] = gender;
        obj["company"] = company;
        obj["weight"] = weight;

        return obj;
    }
};

class dataJson
{
public:
    dataJson() = default;
    ~dataJson() = default;

    QJsonObject toJson() const
    {
        QJsonObject obj;

        obj["name"] = m_name;
        obj["age"] = m_age;
        obj["gender"] = m_gender;
        obj["company"] = m_company;
        obj["weight"] = m_weight;

        return obj;
    }

public:
    const char* m_name;
    int m_age;
    bool m_gender;
    const char* m_company;
    double m_weight;
};

#endif // !__DATA_STRUCT_HPP__
