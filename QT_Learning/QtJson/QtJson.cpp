#include "QtJson.h"
#include <QFile>
#include <QString>
#include <QDebug>
#include <QByteArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStringList>
#include <QStringListModel>

#include "dataStruct.hpp"

QtJson::QtJson(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::QtJsonClass())
{
    ui->setupUi(this);

    connect(ui->readJsonBtn, &QPushButton::clicked, this, &QtJson::sl_ReadJson_Clicked);;
    connect(ui->writeJsonBtn, &QPushButton::clicked, this, &QtJson::sl_WriteJson_Clicked);;

}

QtJson::~QtJson()
{
    delete ui;
}

void QtJson::sl_ReadJson_Clicked()
{
    // 根据路径打开 json 文件
    QString filepatah = R"(./test.json)";
    QFile file(filepatah);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        qDebug() << "打开 Json 文件失败: " << file.errorString();

    // 根据读入的json文件转换为 qt 识别的 JSON
    QByteArray jsonData = file.readAll();
    QJsonDocument jsonDoc = QJsonDocument::fromJson(jsonData);
    QJsonObject jsonObject = jsonDoc.object();

    // 解析 Json 数据格式到具体的数据类型
    QString name = jsonObject["name"].toString();
    int age = jsonObject["age"].toInt();
    bool gender = jsonObject["gender"].toBool();
    QString company = jsonObject["company"].toString();
    double weight = jsonObject["weight"].toDouble();

    // 显示解析的 Json 数据
    QString res1 = QString("name: %1").arg(name);
    QString res2 = QString("age: %1").arg(age);
    QString res3 = QString("gender: %1").arg(gender ? "true" : "false");
    QString res4 = QString("company: %1").arg(company);
    QString res5 = QString("weight: %1").arg(weight);

    QStringList res;
    res.append(res1);
    res.append(res2);
    res.append(res3);
    res.append(res4);
    res.append(res5);

    QStringListModel* listmodel = new QStringListModel(res);
    ui->listView->setModel(listmodel);
    ui->listView->show();
}

void QtJson::sl_WriteJson_Clicked()
{
    /*dataJson data;
    data.m_name = "Li Wei";
    data.m_age = 24;
    data.m_company = "weili.json";
    data.m_gender = false;
    data.m_weight = 88.8;*/

    dataStruct data;
    data.name = "Li Wei";
    data.age = 25;
    data.company = "weili.json";
    data.gender = true;
    data.weight = 88.8;

    // 将需要保存的结构体数据转为 QT Json 数据
    QJsonObject jsonObject = data.toJson();
    QJsonDocument jsonDoc(jsonObject);

    // 根据路径打开 json 文件进行写入数据
    //QString filepatah = R"(./test_write.json)";
    QString filepatah = R"(./test.json)";
    QFile file(filepatah);

    if (file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        file.write(jsonDoc.toJson());
        file.close();
    }
    else
    {
        qDebug() << "写入 Json 文件失败\n";
    }
}
