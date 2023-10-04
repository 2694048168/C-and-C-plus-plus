#pragma once

#include <QObject>
#include <QtPlugin>

#include "affineplugin_global.h"

class AffinePlugin : public QObject, public EditorPluginInterface
{
private:
	Q_OBJECT
		Q_PLUGIN_METADATA(IID EDIT_PLUGIN_INTERFACE_IID);
	Q_INTERFACES(EditorPluginInterface);

public:
	QString name();
	void edit(const cv::Mat& input, cv::Mat& output);
};
