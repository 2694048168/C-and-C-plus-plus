#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QKeyEvent>
#include <QDebug>
#include <QMap.h>
#include <QPluginLoader>

#include <opencv2/opencv.hpp>

#include "mainwindow.h"
#include "editor_plugin_interface.h"


MainWindow::MainWindow(QWidget* parent) :
	QMainWindow(parent)
	, fileMenu(nullptr)
	, viewMenu(nullptr)
	, currentImage(nullptr)
{
	initUI();
	loadPlugins();
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
	editMenu = this->menuBar()->addMenu("&Edit");

	// setup toolbar
	fileToolBar = this->addToolBar("File");
	viewToolBar = this->addToolBar("View");
	editToolBar = this->addToolBar("Edit");

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

	// bluring the image
	blurAction = new QAction("Blur", this);
	editMenu->addAction(blurAction);
	editToolBar->addAction(blurAction);

	// connect the signals and slots
	connect(exitAction, SIGNAL(triggered(bool)), QApplication::instance(), SLOT(quit()));
	connect(openAction, SIGNAL(triggered(bool)), this, SLOT(openImage()));
	connect(zoomInAction, SIGNAL(triggered(bool)), this, SLOT(zoomIn()));
	connect(zoomOutAction, SIGNAL(triggered(bool)), this, SLOT(zoomOut()));
	connect(saveAsAction, SIGNAL(triggered(bool)), this, SLOT(saveAs()));
	connect(prevAction, SIGNAL(triggered(bool)), this, SLOT(prevImage()));
	connect(nextAction, SIGNAL(triggered(bool)), this, SLOT(nextImage()));
	connect(blurAction, SIGNAL(triggered(bool)), this, SLOT(blurImage()));

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

void MainWindow::blurImage()
{
	// dummy implementation
	//qDebug() << "Blurring the image!";

	if (currentImage == nullptr)
	{
		QMessageBox::information(this, "Information", "No image to edit.");
		return;
	}

	// convert the QPixmap into a QImage, construct a cv::Mat using QImage,
	// blur the cv::Mat, then convert cv::Mat back to QImage and QPixmap, respectively.
	QPixmap pixmap = currentImage->pixmap();
	QImage image = pixmap.toImage();
	// an important step,
	image = image.convertToFormat(QImage::Format_RGB888);

	// cv::Mat constructor, color order: BGR(OpenCV) && RGB(QT)?
	//cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
	cv::Mat img_matrix = cv::Mat(
		image.height(),
		image.width(),
		CV_8UC3,
		image.bits(),
		image.bytesPerLine());

	// blur image vai OpenCV library
	cv::Mat img_blur;
	cv::blur(img_matrix, img_blur, cv::Size(3, 3));
	img_matrix = img_blur;

	QImage img_blurred(
		img_matrix.data,
		img_matrix.cols,
		img_matrix.rows,
		img_matrix.step,
		QImage::Format_RGB888);

	pixmap = QPixmap::fromImage(img_blurred);

	// show the blurred image
	imageScene->clear();
	imageView->resetMatrix();
	currentImage = imageScene->addPixmap(pixmap);
	imageScene->update();
	imageView->setSceneRect(pixmap.rect());

	// the buttom status bar infomation update,
	QString status = QString("(editted blur image), %1x%2").arg(pixmap.width()).arg(pixmap.height());
	mainStatusLabel->setText(status);
}

void MainWindow::loadPlugins()
{
	// generate the plugins directory
	QDir pluginsDir(QApplication::instance()->applicationDirPath() + "/plugins");

	// the plugins are library files on different Operatoring System
	QStringList nameFilters;
	nameFilters << "*.so" << "*.dylib" << "*.dll";

	// iterate over that list to try and load	each plugin
	QFileInfoList plugins = pluginsDir.entryInfoList(nameFilters, QDir::NoDotAndDotDot | QDir::Files, QDir::Name);
	foreach(QFileInfo plugin, plugins)
	{
		// Loading one plugin
		QPluginLoader pluginLoader(plugin.absoluteFilePath(), this);

		// a pointer to our plugin interface type
		EditorPluginInterface* plugin_ptr = dynamic_cast<EditorPluginInterface*>(pluginLoader.instance());
		if (plugin_ptr)
		{
			// MainWindow add corrsponding plugin,
			QAction* action = new QAction(plugin_ptr->name());
			editMenu->addAction(action);
			editToolBar->addAction(action);

			// register the loaded plugin in editPlugins map
			editPlugins[plugin_ptr->name()] = plugin_ptr;

			// connect a slot with the action, in the loop??? so need invoke
			connect(action, SIGNAL(triggered(bool)), this, SLOT(pluginPerform()));
			// pluginLoader.unload();
		}
		else
		{
			qDebug() << "bad plugin: " << plugin.absoluteFilePath();
		}
	}
}

void MainWindow::pluginPerform()
{
	if (currentImage == nullptr)
	{
		QMessageBox::information(this, "Information", "No image to edit.");
		return;
	}

	//  the action it just triggered so that it sends the signal and 
	// invokes the slot by calling the sender() function from QT library
	QAction* active_action = qobject_cast<QAction*>(sender());
	EditorPluginInterface* plugin_ptr = editPlugins[active_action->text()];
	if (!plugin_ptr)
	{
		QMessageBox::information(this, "Information", "No plugin is found.");
		return;
	}

	// the same code Aggregation to be here!!!
	QPixmap pixmap = currentImage->pixmap();
	QImage image = pixmap.toImage();
	image = image.convertToFormat(QImage::Format_RGB888);

	cv::Mat mat = cv::Mat(
		image.height(),
		image.width(),
		CV_8UC3,
		image.bits(),
		image.bytesPerLine());

	// call the interface to finish different code,
	plugin_ptr->edit(mat, mat);

	QImage image_edited(
		mat.data,
		mat.cols,
		mat.rows,
		mat.step,
		QImage::Format_RGB888);

	pixmap = QPixmap::fromImage(image_edited);

	// display image by QT
	imageScene->clear();
	imageView->resetMatrix();
	currentImage = imageScene->addPixmap(pixmap);
	imageScene->update();
	imageView->setSceneRect(pixmap.rect());

	// the bottom status bar infomation
	QString status = QString("(editted image), %1x%2").arg(pixmap.width()).arg(pixmap.height());
	mainStatusLabel->setText(status);
}
