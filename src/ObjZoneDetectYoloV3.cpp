#include<iomanip>
#include <iosfwd>
#include<iostream>
#include<opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include <fstream>
#include <string>
#include<dirent.h>
#include<sys/stat.h>
#include<sys/types.h>

#include "IObjZoneDetect.h"

#ifndef CPU_ONLY
    #define GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"
#endif

#ifdef USE_CUDNN
    #define CUDNN
    #include "cudnn.h"
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
        virtual void detect(const cv::Mat& img, vector<Object> &objs, const float confidence_threshold);
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
    void YoloV3Detector::detect(const cv::Mat& img, vector<Object> &objs, const float confidence_threshold)
    {
        Mat im_resize;
        cv::resize(img,im_resize,Size(net->w,net->h));
        image im = ipl_into_image(im_resize);
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
//            if(index==0)
//                obj.cls = 0;
//            else if(index==2 || index==5 || index==6 || index==7)
//                obj.cls = 1;
//            else
//                continue;
            obj.cls = index;
            obj.zone.x = (b.x-b.w/2.)*img.cols;
            obj.zone.y = (b.y-b.h/2.)*img.rows;
            obj.zone.width = (b.x+b.w/2.)*img.cols - obj.zone.x;
            obj.zone.height = (b.y+b.h/2.)*img.rows - obj.zone.y;

            if(obj.zone.x<0) obj.zone.x = 0;
            if(obj.zone.y<0) obj.zone.y = 0;
            if(obj.zone.x+obj.zone.width>=img.cols) obj.zone.width = img.cols - obj.zone.x;
            if(obj.zone.y+obj.zone.height>=img.rows) obj.zone.height = img.rows - obj.zone.y;


            obj.score = score;

            objs.push_back(obj);

        }

    }

    IObjZoneDetect *CreateObjZoneYoloV3Detector(const std::string& cfg_file, const std::string& weights_file,const int gpu_id)
    {
        IObjZoneDetect *detector = new YoloV3Detector(cfg_file,weights_file,gpu_id);
        return detector;
    }
}

