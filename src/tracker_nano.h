#pragma once

#include "opencv2/core/mat.hpp"
#include <string>

namespace my {

/** @brief the Nano tracker is a super lightweight dnn-based general object tracking.
 *
 *  Nano tracker is much faster and extremely lightweight due to special model structure, the whole model size is about 1.9 MB.
 *  Nano tracker needs two models: one for feature extraction (backbone) and the another for localization (neckhead).
 *  Model download link: https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2
 *  Original repo is here: https://github.com/HonglinChu/NanoTrack
 *  Author: HongLinChu, 1628464345@qq.com
 */
class TrackerNano
{
protected:
    TrackerNano() {};  // use ::create()
public:
    virtual ~TrackerNano() {} 

    struct Params
    {
        Params();
        std::string backbone;
        std::string neckhead;
        int backend;
        int target;
    };

    /** @brief Constructor
    @param parameters NanoTrack parameters TrackerNano::Params
    */
    static cv::Ptr<TrackerNano> create(const TrackerNano::Params& parameters = TrackerNano::Params());

    /** @brief Return tracking score
    */
    virtual float getTrackingScore() = 0;

    virtual void init(cv::InputArray image, const cv::Rect& boundingBox) = 0;
    virtual bool update(cv::InputArray image, cv::Rect& boundingBox) = 0;
};

} // my
