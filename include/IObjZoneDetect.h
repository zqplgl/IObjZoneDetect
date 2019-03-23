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
        bool operator ==(Object const& t) const
        {
            return this->zone==t.zone && this->cls==t.cls;
        }
        bool operator !=(Object const& t) const
        {
            return this->zone!=t.zone && this->cls!=t.cls;
        }
    };

    class IObjZoneDetect
    {
        public:
            virtual void detect(const cv::Mat& img, vector<Object> &objs, const float confidence_threshold)=0;
            virtual ~IObjZoneDetect(){}
    };


    IObjZoneDetect *CreateObjZoneYoloV3Detector(const std::string& cfg_file, const std::string& weights_file,const int gpu_id=0);
    IObjZoneDetect *CreateObjZoneMTcnnDetector(const vector<string>& prototxts_file, const vector<string>& weights_file,const int gpu_id=0);
    IObjZoneDetect *CreateObjZoneSSDDetector(const string &deploy_file, const string& weights_file,
                                             const vector<float>& mean_values,const float normal_val, const int gpu_id=0);
}
#endif
