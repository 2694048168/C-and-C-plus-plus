/**
 * @file AddParameter.h
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <QComboBox>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QList>
#include <QMap>
#include <QPushButton>
#include <QTableWidget>
#include <QVariant>
#include <QWidget>

// 参数结构体
struct Parameter
{
    QString  name        = "";
    QString  type        = "double";
    QVariant value       = QVariant();
    QString  description = "";
};

class AddParamWidget : public QWidget
{
    Q_OBJECT

public:
    QVariant         getParameterValue(const QString &name) const;
    bool             setParameterValue(const QString &name, const QVariant &value);
    QList<Parameter> getAllParameters() const;

    void saveToFile(const QString &filename);
    void loadFromFile(const QString &filename);

private slots:
    void addParameterRow();
    void removeSelectedRow();
    void saveParameters();

private:
    void setupUI();
    void createTableRow(const QString &name = "", const QString &type = "double");

private:
    QTableWidget    *mpTable;
    QPushButton     *mpAddButton;
    QPushButton     *mpSaveButton;
    QList<Parameter> mParameters;

public:
    explicit AddParamWidget(QWidget *parent = nullptr);
    ~AddParamWidget();
};
