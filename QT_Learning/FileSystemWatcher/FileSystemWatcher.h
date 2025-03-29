#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_FileSystemWatcher.h"
#include "Watcher.h"

QT_BEGIN_NAMESPACE
namespace Ui { class FileSystemWatcherClass; };
QT_END_NAMESPACE

class FileSystemWatcherWidget : public QMainWindow
{
    Q_OBJECT

public:
    FileSystemWatcherWidget(QWidget* parent = nullptr);
    ~FileSystemWatcherWidget();

slots:
    void sl_AddFilePath();
    void sl_AddFolderPath();

private:
    Ui::FileSystemWatcherClass* ui;

    FileSystemWatcher* mpWatcher;
};
