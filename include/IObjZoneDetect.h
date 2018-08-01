#ifndef IOBJZONEDETECTFASTER_H
#define IOBJZONEDETECTFASTER_H

#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

using namespace std;

namespace ObjZoneDetect
{

    struct Object
    {
        cv::Rect zone;
        float score;
        int cls;
    };

    class IObjZoneDetect
    {
        public:
            virtual void Detect(const cv::Mat& img, vector<Object> &objs, const float confidence_threshold)=0;
            virtual ~IObjZoneDetect(){}
    };


    IObjZoneDetect *CreateObjZoneYoloV3Detector(const std::string& cfg_file, const std::string& weights_file,const int gpu_id);
}
#endif
