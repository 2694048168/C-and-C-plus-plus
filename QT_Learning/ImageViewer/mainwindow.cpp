#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QKeyEvent>
#include <QDebug>

#include "mainwindow.h"

MainWindow::MainWindow(QWidget* parent) :
	QMainWindow(parent)
	, fileMenu(nullptr)
	, viewMenu(nullptr)
	, currentImage(nullptr)
{
	initUI();
}

MainWindow::~MainWindow()
{
}

void MainWindow::initUI()
{
	this->resize(800, 600);
	// steup menubar
	fileMenu = this->menuBar()->addMenu("&File");
	viewMenu = this->menuBar()->addMenu("&View");

	// setup toolbar
	fileToolBar = this->addToolBar("File");
	viewToolBar = this->addToolBar("View");

	// main area for image display
	imageScene = new QGraphicsScene(this);
	imageView = new QGraphicsView(imageScene);
	this->setCentralWidget(imageView);

	// setup status bar
	mainStatusBar = statusBar();
	mainStatusLabel = new QLabel(mainStatusBar);
	mainStatusBar->addPermanentWidget(mainStatusLabel);
	mainStatusLabel->setText("Image Information will be here!");

	//openAction = new QAction("&Open", this);
	//fileMenu->addAction(openAction);
	//fileToolBar->addAction(openAction);
	createActions();
}

void MainWindow::createActions()
{
	// create actions, add them to menus
	openAction = new QAction("&Open", this);
	fileMenu->addAction(openAction);
	saveAsAction = new QAction("&Save as", this);
	fileMenu->addAction(saveAsAction);
	exitAction = new QAction("E&xit", this);
	fileMenu->addAction(exitAction);

	zoomInAction = new QAction("Zoom in", this);
	viewMenu->addAction(zoomInAction);
	zoomOutAction = new QAction("Zoom Out", this);
	viewMenu->addAction(zoomOutAction);
	prevAction = new QAction("&Previous Image", this);
	viewMenu->addAction(prevAction);
	nextAction = new QAction("&Next Image", this);
	viewMenu->addAction(nextAction);

	// add actions to toolbars
	fileToolBar->addAction(openAction);
	viewToolBar->addAction(zoomInAction);
	viewToolBar->addAction(zoomOutAction);
	viewToolBar->addAction(prevAction);
	viewToolBar->addAction(nextAction);

	// connect the signals and slots
	connect(exitAction, SIGNAL(triggered(bool)), QApplication::instance(), SLOT(quit()));
	connect(openAction, SIGNAL(triggered(bool)), this, SLOT(openImage()));
	connect(zoomInAction, SIGNAL(triggered(bool)), this, SLOT(zoomIn()));
	connect(zoomOutAction, SIGNAL(triggered(bool)), this, SLOT(zoomOut()));
	connect(saveAsAction, SIGNAL(triggered(bool)), this, SLOT(saveAs()));
	connect(prevAction, SIGNAL(triggered(bool)), this, SLOT(prevImage()));
	connect(nextAction, SIGNAL(triggered(bool)), this, SLOT(nextImage()));

	// shortcut
	setupShortcuts();
}

void MainWindow::openImage()
{
	//qDebug() << "Slot openImage is called.\n";
	// implement the function of opening an image from disk
	QFileDialog dialog(this);
	dialog.setWindowTitle("Open Image");
	dialog.setFileMode(QFileDialog::ExistingFile);
	dialog.setNameFilter(tr("Images (*.png *.bmp *.jpg)"));

	QStringList filePaths;
	if (dialog.exec())
	{
		filePaths = dialog.selectedFiles();
		showImage(filePaths.at(0));
	}
}

void MainWindow::showImage(QString path)
{
	// crease buffer and some settings 
	imageScene->clear();
	imageView->resetMatrix();

	QPixmap image(path);
	currentImage = imageScene->addPixmap(image);
	imageScene->update();
	imageView->setSceneRect(image.rect());

	// mainwindow the status bar infomation
	QString status = QString("%1, %2x%3, %4 Bytes").arg(path).arg(image.width())
		.arg(image.height()).arg(QFile(path).size());

	mainStatusLabel->setText(status);

	currentImagePath = path;
}

void MainWindow::zoomIn()
{
	imageView->scale(1.2, 1.2);
}

void MainWindow::zoomOut()
{
	imageView->scale(1 / 1.2, 1 / 1.2);
}

void MainWindow::saveAs()
{
	// check the 'current image'
	if (currentImage == nullptr)
	{
		QMessageBox::information(this, "Information", "Nothing to save.");
		return;
	}

	// create dialog to save image file
	QFileDialog dialog(this);
	dialog.setWindowTitle("Save Image As ...");
	dialog.setFileMode(QFileDialog::AnyFile);
	dialog.setAcceptMode(QFileDialog::AcceptSave);
	dialog.setNameFilter(tr("Images (*.png *.bmp *.jpg)"));

	// save image file to the disk
	QStringList fileNames;
	if (dialog.exec())
	{
		fileNames = dialog.selectedFiles();
		if (QRegExp(".+\\.(png|bmp|jpg)").exactMatch(fileNames.at(0)))
		{
			currentImage->pixmap().save(fileNames.at(0));
		}
		else
		{
			QMessageBox::information(this, "Information", "Save error: bad format or filename.");
		}
	}
}

// decide to count the images in alphabetical order according to their names.
// With these two pieces of information, 
// we can now determine which is the previous or next image.
void MainWindow::prevImage()
{
	QFileInfo current(currentImagePath);
	QDir dir = current.absoluteDir();

	QStringList nameFilters;
	nameFilters << "*.png" << "*.bmp" << "*.jpg";
	// list is sorted by filename in alphabetical order.
	QStringList fileNames = dir.entryList(nameFilters, QDir::Files, QDir::Name);

	int idx = fileNames.indexOf(QRegExp(QRegExp::escape(current.fileName())));
	if (idx > 0) 
	{
		showImage(dir.absoluteFilePath(fileNames.at(idx - 1)));
	}
	else 
	{
		QMessageBox::information(this, "Information", "Current image is the first one.");
	}
}

void MainWindow::nextImage()
{
	QFileInfo current(currentImagePath);
	QDir dir = current.absoluteDir();

	QStringList nameFilters;
	nameFilters << "*.png" << "*.bmp" << "*.jpg";
	QStringList fileNames = dir.entryList(nameFilters, QDir::Files, QDir::Name);

	int idx = fileNames.indexOf(QRegExp(QRegExp::escape(current.fileName())));
	if (idx < fileNames.size() - 1) 
	{
		showImage(dir.absoluteFilePath(fileNames.at(idx + 1)));
	}
	else 
	{
		QMessageBox::information(this, "Information", "Current image is the last one.");
	}
}

void MainWindow::setupShortcuts()
{
	QList<QKeySequence> shortcuts;

	// + or =
	shortcuts << Qt::Key_Plus << Qt::Key_Equal;
	// bind the shortcut to slot function correspondingly
	zoomInAction->setShortcuts(shortcuts);

	shortcuts.clear();
	// - or _
	shortcuts << Qt::Key_Minus << Qt::Key_Underscore;
	zoomOutAction->setShortcuts(shortcuts);

	shortcuts.clear();
	shortcuts << Qt::Key_Up << Qt::Key_Left;
	prevAction->setShortcuts(shortcuts);

	shortcuts.clear();
	shortcuts << Qt::Key_Down << Qt::Key_Right;
	nextAction->setShortcuts(shortcuts);
}
