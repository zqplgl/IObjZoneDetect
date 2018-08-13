#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <IObjZoneDetect.h>
#include <bits/shared_ptr.h>

namespace ObjZoneDetect
{
    using namespace std;
    using namespace cv;
    using namespace caffe;

    class MTcnnDetector: public IObjZoneDetect
    {
    public:
        MTcnnDetector(const vector<string>& prototxt_files, const vector<string>& weights_file,const int gpu_id);
        virtual void Detect(const cv::Mat& im, vector<Object> &objs, const float confidence_threshold);

    private:
        cv::Mat im_convert(const cv::Mat& im);
        void im_resize(const cv::Mat&im,vector<Mat>& ims);
        void P_Net(const vector<Mat>& ims);

    private:
        int gpu_id;
        vector<std::shared_ptr<Net<float> > > nets_;
        vector<cv::Size> input_genometry_;
        int min_size_ = 20;
        float factor_ = 0.709;
        int num_channels_ = 3;
    };

    MTcnnDetector::MTcnnDetector(const vector<string> &prototxt_files, const vector<string> &weights_file, const int gpu_id):gpu_id(gpu_id)
    {
#ifndef CPU_ONLY
        Caffe::set_mode(caffe::Caffe::GPU);
        Caffe::SetDevice(this->gpu_id);
#elif
        Caffe::set_mode(Caffe::CPU);
#endif

        for(int i=0; i<prototxt_files.size(); ++i)
        {
            std::shared_ptr<Net<float> > net;
            net.reset(new Net<float>(prototxt_files[i],TEST));
            net->CopyTrainedLayersFrom(weights_file[i]);

            nets_.push_back(net);
        }

        Blob<float>* blob = nets_[0]->input_blobs()[0];
        num_channels_ = blob->channels();
    }

    cv::Mat MTcnnDetector::im_convert(const cv::Mat &im)
    {
        cv::Mat sample;
        if (im.channels() == 3 && num_channels_ == 1)
            cv::cvtColor(im, sample, cv::COLOR_BGR2GRAY);
        else if (im.channels() == 4 && num_channels_ == 1)
            cv::cvtColor(im, sample, cv::COLOR_BGRA2GRAY);
        else if (im.channels() == 4 && num_channels_ == 3)
            cv::cvtColor(im, sample, cv::COLOR_BGRA2BGR);
        else if (im.channels() == 1 && num_channels_ == 3)
            cv::cvtColor(im, sample, cv::COLOR_GRAY2BGR);
        else
            sample = im;

        cv::Mat sample_float;
        if (num_channels_ == 3)
            sample.convertTo(sample_float, CV_32FC3);
        else
            sample.convertTo(sample_float, CV_32FC1);

        cv::cvtColor(sample_float,sample_float,cv::COLOR_BGR2RGB);
        sample_float = sample_float.t();

        return sample_float;
    }

    void MTcnnDetector::im_resize(const cv::Mat &im, vector<cv::Mat> &ims)
    {
        float scale = 12./min_size_;
        int minl = min(im.rows,im.cols);

        while(minl>=12)
        {
            cv::Size size(im.cols*scale,im.rows*scale);
            cv::Mat im_resized;
            cv::resize(im,im_resized,size);

            if(num_channels_==3)
                im_resized.convertTo(im_resized,CV_32FC3,0.0071825,-127.5*0.0071825);
            else
                im_resized.convertTo(im_resized,CV_32FC1,0.0071825,-127.5*0.0071825);
            ims.push_back(im_resized);

            scale *= factor_;
            minl *= factor_;
        }

    }

    void MTcnnDetector::P_Net(const vector<cv::Mat> &ims)
    {
        std::shared_ptr<Net<float> > pnet = nets_[0];
        for(int i=0; i<ims.size(); ++i)
        {

        }
    }

    void MTCNN::Predict(const cv::Mat& img, int i)
    {
        std::shared_ptr<Net> net = nets_[i];
        std::vector<string> output_blob_names = output_blob_names_[i];

        Blob* input_layer = net->blob_by_name("data").get();
        input_layer->Reshape(1, num_channels_,
                             img.rows, img.cols);
        /* Forward dimension change to all layers. */
        net->Reshape();

        std::vector<cv::Mat> input_channels;
        WrapInputLayer(img, &input_channels, i);
        net->Forward();

        /* Copy the output layer to a std::vector */
        Blob* rect = net->blob_by_name(output_blob_names[0]).get();
        Blob* confidence = net->blob_by_name(output_blob_names[1]).get();
        int count = confidence->count() / 2;

        const float* rect_begin = rect->cpu_data();
        const float* rect_end = rect_begin + rect->channels() * count;
        regression_box_temp_ = std::vector<float>(rect_begin, rect_end);

        const float* confidence_begin = confidence->cpu_data() + count;
        const float* confidence_end = confidence_begin + count;

        confidence_temp_ = std::vector<float>(confidence_begin, confidence_end);
    }

    void MTcnnDetector::Detect(const cv::Mat &im, vector<ObjZoneDetect::Object> &objs, const float confidence_threshold)
    {
        cv::Mat im_float = im_convert(im);
        vector<Mat> ims;
        im_resize(im_float,ims);

        for(int i=0; i<ims.size(); ++i)
        {
            nets_[0].
        }
    }

    IObjZoneDetect *CreateObjZoneMTcnnDetector(const vector<string>& prototxts_file, const vector<string>& weights_file,const int gpu_id)
    {
        IObjZoneDetect *detector = new MTcnnDetector(prototxts_file,weights_file,gpu_id);
        return detector;
    }
}

