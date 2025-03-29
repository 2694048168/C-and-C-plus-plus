#pragma once

#include <QObject>
#include <QMap>
#include <QString>
#include <QMap>
#include <QFileSystemWatcher>

class FileSystemWatcher : public QObject
{
    Q_OBJECT

public:
    static void addWatchPath(QString path);

public slots:

    // 目录更新时调用
    void directoryUpdated(const QString& path);

    // 文件被修改时调用
    void fileUpdated(const QString& path);

private:
    explicit FileSystemWatcher(QObject* parent = 0);

private:
    // 单例
    static FileSystemWatcher* m_pInstance;

    // QFileSystemWatcher变量
    QFileSystemWatcher* m_pSystemWatcher;

    // 当前每个监控的内容目录列表
    QMap<QString, QStringList> m_currentContentsMap;
};
