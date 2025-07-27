#pragma once

#include "nlohmann/json.hpp" // 用于 JSON 序列化
#include "opencv2/opencv.hpp"

#include <string>

using json = nlohmann::json;

struct BlobParams
{
    int   minThreshold        = 0;
    int   maxThreshold        = 255;
    int   minArea             = 100;
    int   maxArea             = 10000;
    bool  filterByColor       = true;
    uchar blobColor           = 255;
    bool  filterByCircularity = false;
    float minCircularity      = 0.8f;
    bool  filterByConvexity   = false;
    float minConvexity        = 0.9f;
    bool  filterByInertia     = false;
    float minInertiaRatio     = 0.5f;

    // JSON 序列化
    json toJson() const
    {
        return {
            {       "minThreshold",        minThreshold},
            {       "maxThreshold",        maxThreshold},
            {            "minArea",             minArea},
            {            "maxArea",             maxArea},
            {      "filterByColor",       filterByColor},
            {          "blobColor",           blobColor},
            {"filterByCircularity", filterByCircularity},
            {     "minCircularity",      minCircularity},
            {  "filterByConvexity",   filterByConvexity},
            {       "minConvexity",        minConvexity},
            {    "filterByInertia",     filterByInertia},
            {    "minInertiaRatio",     minInertiaRatio}
        };
    }

    // JSON 反序列化
    void fromJson(const json &j)
    {
        minThreshold        = j.value("minThreshold", 0);
        maxThreshold        = j.value("maxThreshold", 255);
        minArea             = j.value("minArea", 100);
        maxArea             = j.value("maxArea", 10000);
        filterByColor       = j.value("filterByColor", true);
        blobColor           = j.value("blobColor", 255);
        filterByCircularity = j.value("filterByCircularity", false);
        minCircularity      = j.value("minCircularity", 0.8f);
        filterByConvexity   = j.value("filterByConvexity", false);
        minConvexity        = j.value("minConvexity", 0.9f);
        filterByInertia     = j.value("filterByInertia", false);
        minInertiaRatio     = j.value("minInertiaRatio", 0.5f);
    }
};
