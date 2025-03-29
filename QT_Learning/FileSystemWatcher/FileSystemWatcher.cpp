#include "FileSystemWatcher.h"

#include <QFileDialog>

FileSystemWatcherWidget::FileSystemWatcherWidget(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::FileSystemWatcherClass()), mpWatcher{ nullptr }
{
    ui->setupUi(this);

    mpWatcher = new FileSystemWatcher;

    connect(ui->btn_selectFile, &QPushButton::clicked, this, &FileSystemWatcherWidget::sl_AddFilePath);

    connect(ui->btn_selectFolder, &QPushButton::clicked, this, &FileSystemWatcherWidget::sl_AddFolderPath);
}

FileSystemWatcherWidget::~FileSystemWatcherWidget()
{
    delete ui;
    if (mpWatcher)
    {
        delete mpWatcher;
        mpWatcher = nullptr;
    }
}

void FileSystemWatcherWidget::sl_AddFilePath()
{
    QString fileName = QFileDialog::getOpenFileName(
        this,
        tr("open a file."),
        "D:/",
        tr("images(*.png *jpeg *bmp);;video files(*.avi *.mp4 *.wmv);;All files(*.*)"));

    if (fileName.isEmpty())
    {
        QMessageBox::warning(this, "Warning!", "Failed to open the video!");
    }

}

void FileSystemWatcherWidget::sl_AddFolderPath()
{
}
