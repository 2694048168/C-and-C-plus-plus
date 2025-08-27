/**
 * @file AddParameter.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "AddParameter.h"

#include <QComboBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

AddParamWidget::AddParamWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

AddParamWidget::~AddParamWidget()
{
    if (mpTable)
    {
        delete mpTable;
        mpTable = nullptr;
    }

    if (mpAddButton)
    {
        delete mpAddButton;
        mpAddButton = nullptr;
    }

    if (mpSaveButton)
    {
        delete mpSaveButton;
        mpSaveButton = nullptr;
    }

    mParameters.clear();
}

void AddParamWidget::setupUI()
{
    mpTable = new QTableWidget(0, 5, this); // 5列: 名称, 类型, 值, 描述, 操作
    mpTable->setHorizontalHeaderLabels({"Name", "Type", "Value", "Description", "Action"});
    mpTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    mpAddButton  = new QPushButton("Add Parameter", this);
    mpSaveButton = new QPushButton("Save Parameters", this);

    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->addWidget(mpTable);

    QHBoxLayout *buttonLayout = new QHBoxLayout;
    buttonLayout->addWidget(mpAddButton);
    buttonLayout->addWidget(mpSaveButton);
    layout->addLayout(buttonLayout);

    connect(mpAddButton, &QPushButton::clicked, this, &AddParamWidget::addParameterRow);
    connect(mpSaveButton, &QPushButton::clicked, this, &AddParamWidget::saveParameters);
}

void AddParamWidget::addParameterRow()
{
    createTableRow();
}

void AddParamWidget::removeSelectedRow()
{
    QPushButton *button = qobject_cast<QPushButton *>(sender());
    if (button)
    {
        int row = mpTable->indexAt(button->pos()).row();
        if (row >= 0 && row < mpTable->rowCount())
        {
            mpTable->removeRow(row);
        }
    }
}

void AddParamWidget::createTableRow(const QString &name, const QString &type)
{
    const int row = mpTable->rowCount();
    mpTable->insertRow(row);

    // 参数名称输入框
    QLineEdit *nameEdit = new QLineEdit(name);
    mpTable->setCellWidget(row, 0, nameEdit);

    // 数据类型下拉框
    QComboBox *typeCombo = new QComboBox;
    typeCombo->addItems({"int", "double", "float", "QString", "bool"});
    typeCombo->setCurrentText(type);
    mpTable->setCellWidget(row, 1, typeCombo);

    // 值输入框（根据类型变化）
    QLineEdit *valueEdit = new QLineEdit;
    mpTable->setCellWidget(row, 2, valueEdit);

    // 参数描述
    QLineEdit *describeEdit = new QLineEdit;
    mpTable->setCellWidget(row, 3, describeEdit);

    // 删除按钮
    QPushButton *deleteButton = new QPushButton("Delete");
    connect(deleteButton, &QPushButton::clicked, this, &AddParamWidget::removeSelectedRow);
    mpTable->setCellWidget(row, 4, deleteButton);
}

void AddParamWidget::saveParameters()
{
    mParameters.clear();

    for (int row = 0; row < mpTable->rowCount(); ++row)
    {
        QLineEdit *nameEdit     = qobject_cast<QLineEdit *>(mpTable->cellWidget(row, 0));
        QComboBox *typeCombo    = qobject_cast<QComboBox *>(mpTable->cellWidget(row, 1));
        QLineEdit *valueEdit    = qobject_cast<QLineEdit *>(mpTable->cellWidget(row, 2));
        QLineEdit *describeEdit = qobject_cast<QLineEdit *>(mpTable->cellWidget(row, 3));

        if (!nameEdit || !typeCombo || !valueEdit)
            continue;

        QString name        = nameEdit->text().trimmed();
        QString type        = typeCombo->currentText();
        QString valueStr    = valueEdit->text();
        QString describeStr = describeEdit->text();

        if (name.isEmpty())
        {
            QMessageBox::warning(this, "Error", "Parameter name cannot be empty!");
            return;
        }

        // 创建参数并转换值
        Parameter param;

        param.description = describeStr;
        // 根据类型转换值
        if (type == "int")
        {
            bool ok;
            int  val = valueStr.toInt(&ok);
            if (!ok)
            {
                QMessageBox::warning(this, "Error", "Invalid integer value: " + valueStr);
                return;
            }
            param.value = val;
        }
        else if (type == "double")
        {
            bool   ok;
            double val = valueStr.toDouble(&ok);
            if (!ok)
            {
                QMessageBox::warning(this, "Error", "Invalid double value: " + valueStr);
                return;
            }
            param.value = val;
        }
        else if (type == "float")
        {
            bool   ok;
            double val = valueStr.toFloat(&ok);
            if (!ok)
            {
                QMessageBox::warning(this, "Error", "Invalid float value: " + valueStr);
                return;
            }
            param.value = val;
        }
        else if (type == "bool")
        {
            param.value = (valueStr.compare("true", Qt::CaseInsensitive) == 0);
        }
        else
        { // QString
            param.value = valueStr;
        }

        mParameters.append(param);
    }

    QMessageBox::information(this, "Success", "Parameters saved successfully!");
}

QVariant AddParamWidget::getParameterValue(const QString &name) const
{
    for (const Parameter &param : mParameters)
    {
        if (param.name == name)
        {
            return param.value;
        }
    }
    return QVariant(); // 返回无效值
}

bool AddParamWidget::setParameterValue(const QString &name, const QVariant &value)
{
    for (Parameter &param : mParameters)
    {
        if (param.name == name)
        {
            param.value = value;
            return true;
        }
    }
    return false; // 返回无效值
}

QList<Parameter> AddParamWidget::getAllParameters() const
{
    return mParameters;
}

void AddParamWidget::saveToFile(const QString &filename)
{
    QJsonArray jsonArray;

    for (const Parameter &param : mParameters)
    {
        QJsonObject obj;
        obj["name"]        = param.name;
        obj["type"]        = param.type;
        obj["description"] = param.description;

        // 根据类型存储值
        if (param.type == "int")
        {
            obj["value"] = param.value.toInt();
        }
        else if (param.type == "double")
        {
            obj["value"] = param.value.toDouble();
        }
        else if (param.type == "float")
        {
            obj["value"] = param.value.toFloat();
        }
        else if (param.type == "bool")
        {
            obj["value"] = param.value.toBool();
        }
        else
        { // QString
            obj["value"] = param.value.toString();
        }

        jsonArray.append(obj);
    }

    QFile file(filename);
    if (file.open(QIODevice::WriteOnly))
    {
        file.write(QJsonDocument(jsonArray).toJson());
        file.close();
    }
}

void AddParamWidget::loadFromFile(const QString &filename)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly))
        return;

    QJsonArray jsonArray = QJsonDocument::fromJson(file.readAll()).array();
    mpTable->setRowCount(0); // 清空表格
    mParameters.clear();

    for (const QJsonValue &val : jsonArray)
    {
        QJsonObject obj         = val.toObject();
        QString     name        = obj["name"].toString();
        QString     type        = obj["type"].toString();
        QString     description = obj["description"].toString();

        createTableRow(name, type);

        // 创建参数对象
        Parameter param;
        if (type == "int")
        {
            param.value = obj["value"].toInt();
        }
        else if (type == "double")
        {
            param.value = obj["value"].toDouble();
        }
        else if (type == "float")
        {
            param.value = obj["value"].toDouble();
        }
        else if (type == "bool")
        {
            param.value = obj["value"].toBool();
        }
        else
        { // QString
            param.value = obj["value"].toString();
        }

        mParameters.append(param);

        // 设置值显示
        int row = mpTable->rowCount() - 1;
        if (QLineEdit *valueEdit = qobject_cast<QLineEdit *>(mpTable->cellWidget(row, 2)))
        {
            valueEdit->setText(param.value.toString());
        }
    }
}
