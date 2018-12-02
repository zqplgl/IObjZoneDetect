//
// Created by zqp on 18-12-2.
//
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <IObjZoneDetect.h>
#include <bits/shared_ptr.h>

namespace ObjZoneDetect
{
    using namespace std;
    using namespace cv;
    using namespace caffe;

    class SSDDetector: public IObjZoneDetect
    {
    public:
        SSDDetector(const string& deploy_file, const string& weight_file, const vector<float>& mean_values, const float normal_val = 1,const int gpu_id=0);
        virtual void Detect(const cv::Mat& im, vector<Object> &objs, const float confidence_threshold);

    private:
        void setMean(const vector<float>& mean_values);
        void wrapInputLayer(const Mat &im);
        cv::Mat imConvert(const cv::Mat &im);

    private:
        boost::shared_ptr<Net<float> > net_;
        int gpu_id_ = 0;
        int num_channels_;
        int imgtype_;
        float nor_val_ = 1;
        cv::Size input_geometry_;
        Mat mean_;
    };

    SSDDetector::SSDDetector(const string &deploy_file,
            const string &weight_file,
            const vector<float> &mean_values,
            const float normal_val,
            const int gpu_id)
    {
        if(gpu_id<0)
            Caffe::set_mode(Caffe::CPU);
        else
        {
            Caffe::set_mode(Caffe::GPU);
            Caffe::SetDevice(gpu_id);
            this->gpu_id_ = gpu_id;
        }

//Load network
        net_.reset(new Net<float>(deploy_file,TEST));
        net_->CopyTrainedLayersFrom(weight_file);

        CHECK_EQ(net_->num_inputs(),1) << "Network should have exactly one input.";
        CHECK_EQ(net_->num_outputs(),1) << "Network should have exactly one output.";

        Blob<float>* input_layer = net_->input_blobs()[0];
        num_channels_ = input_layer->channels();

        CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";

        input_geometry_ = cv::Size(input_layer->width(),input_layer->height());

        setMean(mean_values);
        nor_val_ = normal_val;
   }

   void SSDDetector::setMean(const vector<float> &mean_values)
   {
       CHECK_EQ(num_channels_,mean_values.size())<<"error mean_values.";
       std::vector<cv::Mat> channels;
       for (int i = 0; i < num_channels_; ++i) {
           /* Extract an individual channel. */
           cv::Mat channel(input_geometry_, CV_32FC1, cv::Scalar(mean_values[i]));
           channels.push_back(channel);
       }
       if (num_channels_==3)
           cv::merge(channels, mean_);
       else
           mean_ = channels[0];
   }

   void SSDDetector::Detect(const cv::Mat &im, vector<ObjZoneDetect::Object> &objs, const float confidence_threshold)
   {
        objs.clear();
       cv::Mat im_normalized = imConvert(im);
       wrapInputLayer(im_normalized);

       net_->Forward();

       Blob<float>* result_blob = net_->output_blobs()[0];
       const float* result = result_blob->cpu_data();
       const int height = result_blob->height();
       const int width = result_blob->width();
       const int channel = result_blob->channels();
       const int num = result_blob->num();
       for (int k = 0; k < height; ++k)
       {
           // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
           vector<float> detection(result,result+7);
           if (result[0] == -1 || result[2]<confidence_threshold)
           {
               result += 7;
               continue;
           }

           int x1 = static_cast<int>(result[3]*im.cols);
           int y1 = static_cast<int>(result[4]*im.rows);
           int x2 = static_cast<int>(result[5]*im.cols);
           int y2 = static_cast<int>(result[6]*im.rows);

           if(x1<0) x1 = 0;
           if(y1<0) y1=0;
           if(x2>=im.cols) x2 = im.cols - 1;
           if(y2>=im.rows) y2 = im.rows - 1;

           ObjZoneDetect::Object obj;
           obj.cls = static_cast<int>(result[1]);
           obj.score = result[2];
           obj.zone = cv::Rect(x1,y1,x2-x1+1,y2-y1+1);
           result += 7;

           objs.push_back(obj);
       }
   }

    cv::Mat SSDDetector::imConvert(const cv::Mat &im)
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

        cv::Mat sample_resized;
        if (sample.size() != input_geometry_)
            cv::resize(sample, sample_resized, input_geometry_);
        else
            sample_resized = sample;

        cv::Mat sample_float;
        if (num_channels_ == 3)
            sample_resized.convertTo(sample_float, CV_32FC3,nor_val_);
        else
            sample_resized.convertTo(sample_float, CV_32FC1,nor_val_);

        cv::Mat sample_normalized;
        cv::subtract(sample_float,mean_,sample_normalized);

        return sample_normalized;
    }

    void SSDDetector::wrapInputLayer(const Mat &im)
    {
        vector<Mat> input_channels;
        input_channels.clear();
        Blob<float>* input_layer = net_->input_blobs()[0];
        input_layer->Reshape(1,num_channels_,im.rows,im.cols);
        net_->Reshape();

        int width = input_layer->width();
        int height = input_layer->height();
        float* input_data = input_layer->mutable_cpu_data();

        int size = width * height;
        for (int i = 0; i < num_channels_; ++i)
        {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += size;
        }

        cv::split(im,input_channels);

        assert(reinterpret_cast<float*>(input_channels.at(0).data)==net_->input_blobs()[0]->cpu_data());
    }

    IObjZoneDetect *CreateObjZoneSSDDetector(const string &deploy_file, const string& weights_file,
            const vector<float>& mean_values,const float normal_val, const int gpu_id)
    {
        IObjZoneDetect *detector = new SSDDetector(deploy_file, weights_file, mean_values, normal_val, gpu_id);
        return detector;
    }

}


