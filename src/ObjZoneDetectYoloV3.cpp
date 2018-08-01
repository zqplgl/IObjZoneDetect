#include<iomanip>
#include <iosfwd>
#include<iostream>
//#include<opencv2/opencv.hpp>
#include <time.h>
#include <fstream>
#include <string>
#include<dirent.h>
#include<sys/stat.h>
#include<sys/types.h>

#include "IObjZoneDetect.h"

#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

extern "C"
{
#include "darknet.h"
}

namespace ObjZoneDetect
{
    using namespace std;
    using namespace cv;

    class YoloV3Detector: public IObjZoneDetect
    {
    public:
        YoloV3Detector(const string& cfg_file, const string& weights_file,const int gpu_id);
        virtual void Detect(const cv::Mat& img, vector<Object> &objs, const float confidence_threshold);
        ~YoloV3Detector();

    private:
        image ipl_into_image(const Mat &src);
        int getArgIndex(const float *prob,int n,float &cls);
        int gpu_id;
        network *net;
    };

    YoloV3Detector::YoloV3Detector(const string& cfg_file, const string& weights_file,const int gpu_id):gpu_id(gpu_id)
    {
        char cfgfile[512];
        strcpy(cfgfile,cfg_file.c_str());

        char weightsfile[512];
        strcpy(weightsfile,weights_file.c_str());
        net = load_network(cfgfile,weightsfile,0);
        set_batch_network(net,1);
        srand(2222222);
    }

    YoloV3Detector::~YoloV3Detector()
    {
        delete(net);
    }

    image YoloV3Detector::ipl_into_image(const Mat &src)
    {
        uchar *data = (uchar *)src.data;
        int w = src.cols;
        int h = src.rows;
        int c = src.channels();

        image im = make_image(w,h,c);
        int step = src.cols*src.channels();
        int i, j, k;

        for(i = 0; i < h; ++i)
            for(k= 0; k < c; ++k)
                for(j = 0; j < w; ++j)
                    im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.f;

        return im;
    }

    int YoloV3Detector::getArgIndex(const float *prob,int n,float &cls)
    {
        int r;
        for(int i=0; i<n; ++i)
        {
            if(cls<prob[i])
            {
                cls = prob[i];
                r = i;
            }
        }

        return r;
    }
    void YoloV3Detector::Detect(const cv::Mat& img, vector<Object> &objs, const float confidence_threshold)
    {
        image im = ipl_into_image(img);
        image sized = letterbox_image(im,net->w,net->h);
        layer l = net->layers[net->n-1];
        network_predict(net,sized.data);
        int nboxes = 0;

        float nms = .45;
        detection *dets = get_network_boxes(net,im.w,im.h,confidence_threshold,0.5,0,1,&nboxes);
        if(nms)do_nms_sort(dets,nboxes,l.classes,nms);

        objs.clear();

        for(int i=0; i<nboxes; ++i)
        {
            float score = -1;
            int index = getArgIndex(dets[i].prob,l.classes,score);

            if(score<confidence_threshold)
                continue;

            box &b = dets[i].bbox;
            Object obj;
            obj.zone.x = (b.x-b.w/2.)*im.w;
            obj.zone.y = (b.y-b.h/2.)*im.h;
            obj.zone.width = (b.x+b.w/2.)*im.w - obj.zone.x;
            obj.zone.height = (b.y+b.h/2.)*im.h - obj.zone.y;

            obj.score = score;
            obj.cls = index;

            objs.push_back(obj);

        }

    }

    IObjZoneDetect *CreateObjZoneYoloV3Detector(const std::string& cfg_file, const std::string& weights_file,const int gpu_id)
    {
        IObjZoneDetect *detector = new YoloV3Detector(cfg_file,weights_file,gpu_id);
        return detector;
    }
}

