#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <IObjZoneDetect.h>

namespace ObjZoneDetect
{
    using namespace std;
    using namespace cv;
    using namespace caffe;

    class MTcnnDetector: public IObjZoneDetect
    {
    public:
        MTcnnDetector(const vector<string>& prototxt_files, const vector<string>& weights_file,const int gpu_id);
        virtual void Detect(const cv::Mat& img, vector<Object> &objs, const vector<float>&  confidence_threshold);
        ~MTcnnDetector();

    private:
        int gpu_id;
    };

    MTcnnDetector::MTcnnDetector(const vector<string> &prototxt_files, const vector<string> &weights_file,
                                 const int gpu_id):gpu_id(gpu_id) {

#ifndef CPU_ONLY
        Caffe::set_mode(caffe::Caffe::GPU);
        Caffe::SetDevice(this->gpu_id);

#elif
        Caffe::set_mode(Caffe::CPU);

#endif

    }

    void MTcnnDetector::Detect(const cv::Mat &img, vector<ObjZoneDetect::Object> &objs,
                               const vector<float> &confidence_threshold) {

  }
    //IObjZoneDetect *CreateObjZoneYoloV3Detector(const std::string& cfg_file, const std::string& weights_file,const int gpu_id)
    //{
    //    IObjZoneDetect *detector = new YoloV3Detector(cfg_file,weights_file,gpu_id);
    //    return detector;
    //}
}

