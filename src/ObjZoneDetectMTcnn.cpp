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
        cv::Mat imConvert(const cv::Mat& im);
        void imResize(const cv::Mat&im,vector<Mat>& ims,vector<float>& scales);
        void pNet(const vector<Mat>& ims,const vector<float>& scales,vector<vector<float> >& total_boxes);
        void wrapInputLayer(std::vector<cv::Mat>& input_channels,const int i);
        void generateBoundingBox(const Blob<float>* prob,const Blob<float>* reg, const float scale, const float threshold,vector<vector<float> >& boxes);
        void nms(vector<vector<float> >& bboxes,const float threshold);
        float calcDistIOU(const vector<float>& bbox1, const vector<float>& bbox2);

        void addRectangle(cv::Mat &im,const vector<vector<float> >& bboxes);

    private:
        int gpu_id;
        vector<std::shared_ptr<Net<float> > > nets_;
        vector<cv::Size> input_genometry_;
        int min_size_ = 20;
        float factor_ = 0.709;
        int num_channels_ = 3;
        vector<pair<string,string> > out_blobnames_;
        vector<float> thresholds_ = {0.6,0.7,0.7};
        float inner_nms_ = 0.5;
        float outer_nms_ = 0.7;

        cv::Mat im_;
    };

    void printMat(const Mat& im)
    {
        ofstream out("save.txt");
        //const uchar *data = im.data;
        const float *data = (float*)im.data;
        int num = im.rows*im.cols*im.channels();
        for(int i=0; i<num; ++i)
        {
            if(i%10==0)
                out<<endl;
            out<<float(data[i])<<"\t";
        }
    }

    void printBlob(const Blob<float>* blob)
    {
        ofstream out("save.txt");
        const float *data = blob->cpu_data();
        for(int i=0; i<blob->count(); ++i)
        {
            if(i%10==0)
                out<<endl;
            out<<float(data[i])<<"\t";
        }

    }

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

        out_blobnames_.push_back(make_pair<string,string>("prob1","conv4-2"));
        out_blobnames_.push_back(make_pair<string,string>("prob1","conv5-2"));
        out_blobnames_.push_back(make_pair<string,string>("prob1","conv6-2"));
    }

    cv::Mat MTcnnDetector::imConvert(const cv::Mat &im)
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
        im_ = sample_float;

        return im_;
    }

    void MTcnnDetector::wrapInputLayer(std::vector<cv::Mat> &input_channels, const int i)
    {
        input_channels.clear();
        Blob<float>* input_layer = nets_[i]->input_blobs()[0];
        assert(input_layer->channels()==num_channels_);

        int width = input_layer->width();
        int height = input_layer->height();
        float* input_data = input_layer->mutable_cpu_data();

        int size = width * height;
        for (int i = 0; i < input_layer->channels(); ++i)
        {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += size;
        }
    }

    void MTcnnDetector::imResize(const cv::Mat &im, vector<cv::Mat> &ims,vector<float>& scales)
    {
        ims.clear();
        scales.clear();
        float scale = 12./min_size_;
        int minl = min(im.rows,im.cols)*scale;

        while(minl>=12)
        {
            cv::Size size(ceil(im.cols*scale),ceil(im.rows*scale));
            scales.push_back(scale);
            cv::Mat im_resized;
            cv::resize(im,im_resized,size);

            if(num_channels_==3)
                im_resized.convertTo(im_resized,CV_32FC3,0.0078125,-127.5*0.0078125);
            else
                im_resized.convertTo(im_resized,CV_32FC1,0.0078125,-127.5*0.0078125);
            ims.push_back(im_resized);

            scale *= factor_;
            minl *= factor_;
        }

    }

    int argIndex(const vector<vector<float> > &bboxes)
    {
        if (bboxes.size()<1)
            return -1;

        int result = 0;
        for(int i=1; i<bboxes.size(); ++i)
        {
            if(bboxes[result][4]<bboxes[i][4])
                result = i;
        }

        return result;
    }

    float MTcnnDetector::calcDistIOU(const vector<float> &bbox1, const vector<float> &bbox2)
    {
        float x1 = max(bbox1[0],bbox2[0]);
        float y1 = max(bbox1[1],bbox2[1]);
        float x2 = min(bbox1[2],bbox2[2]);
        float y2 = min(bbox1[3],bbox2[3]);
        float area = max(0.f,(x2-x1))*max(0.f,(y2-y1));
        float iou = 0;

        if(area>0)
            iou = area/(((bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1]))+((bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1]))-area);

        return iou;

    }

    void MTcnnDetector::nms(vector<vector<float> > &bboxes,const float threshold)
    {
        vector<vector<float> > keeps;

        while(bboxes.size())
        {
            int arg_index = argIndex(bboxes);
            vector<float> bbox = bboxes[arg_index];
            keeps.push_back(bbox);

            bboxes.erase(bboxes.begin()+arg_index);
            for(int i=0; i<bboxes.size(); ++i)
            {
                if(calcDistIOU(bbox,bboxes[i])>=threshold)
                {
                    bboxes.erase(bboxes.begin()+i);
                    i--;
                }
            }
        }

        for(int i=0; i<keeps.size(); ++i)
        {
            bboxes.push_back(keeps[i]);
        }
    }

    void MTcnnDetector::generateBoundingBox(const caffe::Blob<float> *prob, const caffe::Blob<float> *reg,
            const float scale, const float threshold,vector<vector<float> >& boxes)
    {
        int stride = 2;
        int cellsize = 12;
        boxes.clear();

        int prob_h = prob->height();
        int prob_w = prob->width();
        const float* scores = prob->cpu_data()+prob->count()/2;

        int size = prob_h*prob_w;

        const float* dx1 = reg->cpu_data();
        const float* dy1 = dx1 + size;
        const float* dx2 = dy1 + size;
        const float* dy2 = dx2 + size;

        for(int h=0; h<prob_h; ++h)
        {
            int line = h*prob_w;
            for(int w=0; w<prob_w; ++w)
            {
                if(scores[line+w]<threshold)
                    continue;

                vector<float> bbox;
                float x1 = (w*stride+1)/scale;
                float y1 = (h*stride+1)/scale;
                float x2 = (w*stride+cellsize)/scale;
                float y2 = (h*stride+cellsize)/scale;
                float score = scores[line+w];

                float box_w = x2 - x1;
                float box_h = y2 - y1;
                x1 = x1 + dx1[line+w]*box_w;
                y1 = y1 + dy1[line+w]*box_h;
                x2 = x2 + dx2[line+w]*box_w;
                y2 = y2 + dy2[line+w]*box_h;

                bbox.push_back(y1);
                bbox.push_back(x1);
                bbox.push_back(y2);
                bbox.push_back(x2);
                bbox.push_back(score);

                boxes.push_back(bbox);
            }
        }
        nms(boxes,inner_nms_);
    }

    void MTcnnDetector::pNet(const vector<cv::Mat> &ims,const vector<float>& scales,vector<vector<float> >& total_boxes)
    {
        total_boxes.clear();
        vector<Mat> input_channels;
        std::shared_ptr<Net<float> > pnet = nets_[0];
        pair<string,string> out_blobname = out_blobnames_[0];

        for(int i=0; i<ims.size(); ++i)
        {
            Blob<float>* input_blob = pnet->input_blobs()[0];
            input_blob->Reshape(1,num_channels_,ims[i].cols,ims[i].rows);
            pnet->Reshape();

            wrapInputLayer(input_channels,0);
            cv::split(ims[i].t(), input_channels);
            assert(reinterpret_cast<float*>(input_channels.at(0).data)==pnet->input_blobs()[0]->cpu_data());

            pnet->Forward();

            Blob<float>* prob = pnet->blob_by_name(out_blobname.first).get();
            Blob<float>* reg = pnet->blob_by_name(out_blobname.second).get();
            printBlob(prob);

            vector<vector<float> > boxes;
            generateBoundingBox(prob, reg,scales[i],thresholds_[0],boxes);

            for(int j=0; j<boxes.size(); ++j)
                total_boxes.push_back(boxes[j]);
        }
        nms(total_boxes,outer_nms_);

    }

    void MTcnnDetector::Detect(const cv::Mat &im, vector<ObjZoneDetect::Object> &objs, const float confidence_threshold)
    {
        cv::Mat im_float = imConvert(im);

        vector<Mat> ims;
        vector<float> scales;
        imResize(im_float,ims,scales);
        vector<vector<float> > boxes;

        pNet(ims,scales,boxes);
    }

    void MTcnnDetector::addRectangle(cv::Mat &im, const vector<vector<float> > &bboxes)
    {
        for(int i=0; i<bboxes.size(); ++i)
        {
            rectangle(im,Point(bboxes[i][0],bboxes[i][1]),Point(bboxes[i][2],bboxes[i][3]),Scalar(0,0,255));
        }
    }

    IObjZoneDetect *CreateObjZoneMTcnnDetector(const vector<string>& prototxts_file, const vector<string>& weights_file,const int gpu_id)
    {
        IObjZoneDetect *detector = new MTcnnDetector(prototxts_file,weights_file,gpu_id);
        return detector;
    }
}

