#include "erode_plugin.h"

QString ErodePlugin::name()
{
	return "Erode";
}

void ErodePlugin::edit(const cv::Mat& input, cv::Mat& output)
{
	// the area of mathematical morphology from OpenCV library
	cv::erode(input, output, cv::Mat());
}
