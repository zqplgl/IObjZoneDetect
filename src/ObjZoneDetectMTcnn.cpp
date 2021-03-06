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
        virtual void detect(const cv::Mat& im, vector<Object> &objs, const float confidence_threshold);

    private:
        cv::Mat imConvert(const cv::Mat& im);
        void imResize(const cv::Mat&im,vector<Mat>& ims,vector<float>& scales);
        void pNet(const vector<Mat>& ims,const vector<float>& scales,vector<vector<float> >& total_boxes);
        void rNet(const vector<Mat>& ims,vector<vector<float> >& total_boxes);
        void oNet(const vector<Mat>& ims,vector<vector<float> >& total_boxes);
        void wrapInputLayer(const Mat &im,const int i);
        void generateBoundingBox(const Blob<float>* prob,const Blob<float>* reg, const float scale, const float threshold,vector<vector<float> >& boxes);
        void nms(vector<vector<float> >& bboxes,const float threshold);
        float calcDistIOU(const vector<float>& bbox1, const vector<float>& bbox2);
        void rerac(vector<vector<float> > &boxes);
        void pad(vector<vector<float> > & boxes,const cv::Mat &im,vector<vector<float> >&dst_boxes);
        void wrapInputLayer(const vector<Mat> &ims,const int i);
        void generateRois(const Mat &im, const vector<vector<float> >&boxes,const vector<vector<float> >&dst_boxes,vector<Mat>& ims);
        void bbreg(vector<vector<float> >& boxes,const Blob<float>* prob, const Blob<float>* reg,const float threshold);
        float fix(float a);
        void bbreg(vector<vector<float> >& boxes);

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
            out<<data[i]<<"\t";
        }
    }

    void printBlob(const Blob<float>* blob)
    {
        ofstream out("save.txt");
        const float *data = blob->cpu_data();
        for(int i=0; i<blob->count(); ++i)
        {
            if(i%5==0)
                out<<endl;
            out<<data[i]<<"\t";
        }

    }

    void printBBox(const vector<vector<float> >& bboxes)
    {
        ofstream out("save.txt");
        for(int i=0; i<bboxes.size(); ++i)
        {
            for(int j=0; j<bboxes[i].size(); ++j)
                out<<bboxes[i][j]<<"\t";
            out<<endl;
        }
    }

    void printBBox(const vector<vector<float> >& bboxes1,const vector<vector<float> >& bboxes2)
    {
        ofstream out("save.txt");
        for(int i=0; i<bboxes1.size(); ++i)
        {
            for(int j=0; j<bboxes1[i].size(); ++j)
                out<<bboxes1[i][j]<<"\t";
            for(int j=0; j<4; ++j)
                out<<bboxes2[i][j]<<"\t";
            out<<endl;
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

    float MTcnnDetector::fix(float a)
    {
        float temp = a;
        a = abs(a);
        float result = round(a);
        if(a>0 && (result - a)>5e-5)
            result--;

        return temp>=0.f ? result: -result;
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

    void MTcnnDetector::wrapInputLayer(const Mat &im, const int i)
    {
        vector<Mat> input_channels;
        input_channels.clear();
        Blob<float>* input_layer = nets_[i]->input_blobs()[0];
        input_layer->Reshape(1,num_channels_,im.cols,im.rows);
        nets_[i]->Reshape();

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

        cv::split(im.t(),input_channels);
        assert(reinterpret_cast<float*>(input_channels.at(0).data)==nets_[i]->input_blobs()[0]->cpu_data());
    }

    void MTcnnDetector::wrapInputLayer(const vector<cv::Mat> &ims, const int i)
    {
        vector<Mat> input_channels;
        input_channels.clear();
        Blob<float>* input_layer = nets_[i]->input_blobs()[0];
        assert(input_layer->channels()==num_channels_);

        int width = input_layer->width();
        int height = input_layer->height();
        input_layer->Reshape(ims.size(),num_channels_,height,width);
        nets_[i]->Reshape();

        float* input_data = input_layer->mutable_cpu_data();

        int size = width * height;

        printMat(ims[0]);

        for(int i=0; i<ims.size(); ++i)
        {
            Mat im;
            cv::resize(ims[i],im,Size(height,width));
            im.convertTo(im,CV_32FC3,0.0078125,-127.5*0.0078125);
            input_channels.clear();
            for (int i = 0; i < input_layer->channels(); ++i)
            {
                cv::Mat channel(height, width, CV_32FC1, input_data);
                input_channels.push_back(channel);
                input_data += size;
            }

            split(im.t(),input_channels);
        }
    }

    void MTcnnDetector::imResize(const cv::Mat &im, vector<cv::Mat> &ims,vector<float>& scales)
    {
        ims.clear();
        scales.clear();
        float scale = 12.f/min_size_;
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
        float area = max(0.f,(x2-x1+1))*max(0.f,(y2-y1+1));
        float iou = 0;

        if(area>0)
            iou = area/(((bbox1[2]-bbox1[0]+1)*(bbox1[3]-bbox1[1]+1))+((bbox2[2]-bbox2[0]+1)*(bbox2[3]-bbox2[1]+1))-area);

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
                if(calcDistIOU(bbox,bboxes[i])>threshold)
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

        for(int w=0; w<prob_w; ++w)
        {
            for(int h=0; h<prob_h; ++h)
            {
                int index = h*prob_w + w;
                if(scores[index]<threshold)
                    continue;

                vector<float> bbox;
                float y1 = (w*stride+1)/scale;
                float x1 = (h*stride+1)/scale;
                float y2 = (w*stride+cellsize)/scale;
                float x2 = (h*stride+cellsize)/scale;
                x2 = fix(x2);
                float score = scores[index];

                bbox.push_back(fix(x1));
                bbox.push_back(fix(y1));
                bbox.push_back(fix(x2));
                bbox.push_back(fix(y2));
                bbox.push_back(score);
                bbox.push_back(dx1[index]);
                bbox.push_back(dy1[index]);
                bbox.push_back(dx2[index]);
                bbox.push_back(dy2[index]);

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
            wrapInputLayer(ims[i],0);
            pnet->Forward();

            Blob<float>* prob = pnet->blob_by_name(out_blobname.first).get();
            Blob<float>* reg = pnet->blob_by_name(out_blobname.second).get();
            printBlob(reg);
            vector<vector<float> > boxes;
            generateBoundingBox(prob, reg,scales[i],thresholds_[0],boxes);

            for(int j=0; j<boxes.size(); ++j)
                total_boxes.push_back(boxes[j]);
        }
        nms(total_boxes,outer_nms_);
    }

    void MTcnnDetector::rerac(vector<vector<float> > &boxes)
    {
        for(int i=0; i<boxes.size(); ++i)
        {
            float w = boxes[i][2] - boxes[i][0];
            float h = boxes[i][3] - boxes[i][1];

            if(w==h)
                continue;
            else if(w>h)
            {
                boxes[i][1] -= (w-h)/2;
                boxes[i][3] = boxes[i][1] + w;
            }
            else
            {
                boxes[i][0] -= (h-w)/2;
                boxes[i][2] = boxes[i][0] + h;
            }
            for(int j=0; j<4; ++j)
                boxes[i][j] = fix(boxes[i][j]);

        }
    }

    void MTcnnDetector::pad(vector<vector<float> > & boxes,const cv::Mat &im,vector<vector<float> >&dst_boxes)
    {
        dst_boxes.clear();
        for(int i=0; i<boxes.size(); ++i)
        {
            //x1,y1,x2,y2,side
            vector<float> dst_box;
            float tempw = boxes[i][2] - boxes[i][0] + 1;
            float temph = boxes[i][3] - boxes[i][1] + 1;

            if(boxes[i][0]<1)
            {
                dst_box.push_back(1 - boxes[i][0]);
                boxes[i][0] = 0;
            }
            else
            {
                dst_box.push_back(0);
                boxes[i][0] = max(0.f,boxes[i][0] - 1);
            }
            if(boxes[i][1]<1)
            {
                dst_box.push_back(1 - boxes[i][1]);
                boxes[i][1] = 0;
            }
            else
            {
                dst_box.push_back(0);
                boxes[i][1] = max(0.f,boxes[i][1] - 1);
            }

            if(boxes[i][2]>=im.cols - 1)
            {
                dst_box.push_back(tempw - (boxes[i][2] - (im.cols - 1)) - 1);
                boxes[i][2] = im.cols - 2;
            }
            else
            {
                dst_box.push_back(tempw - 1);
                boxes[i][2] = max(0.f,boxes[i][2] - 1);
            }
            if(boxes[i][3]>=im.rows - 1)
            {
                dst_box.push_back(temph - (boxes[i][3] - (im.rows - 1)) - 1);
                boxes[i][3] = im.rows - 2;
            }
            else
            {
                dst_box.push_back(temph - 1);
                boxes[i][3] = max(0.f,boxes[i][3] - 1);
            }
            dst_box.push_back(tempw);
            dst_box.push_back(temph);

            dst_boxes.push_back(dst_box);
        }

    }

    void MTcnnDetector::generateRois(const Mat &im,const vector<vector<float> > &boxes, const vector<vector<float> > &dst_boxes, vector<cv::Mat> &ims)
    {
        assert(boxes.size()==dst_boxes.size());
        ims.clear();

        for(int i=0; i<dst_boxes.size(); ++i)
        {
            Mat im_roi;
            if(num_channels_==3)
                im_roi = Mat(dst_boxes[i][5] + 1,dst_boxes[i][4] + 1,CV_32FC3,Scalar(0,0,0));
            else
                im_roi = Mat(dst_boxes[i][5],dst_boxes[i][4],CV_32FC1,Scalar(0));

            im(Range(boxes[i][1],boxes[i][3]+1),Range(boxes[i][0],boxes[i][2]+1)). \
                copyTo(im_roi(Range(dst_boxes[i][1],dst_boxes[i][3]+1),Range(dst_boxes[i][0],dst_boxes[i][2]+1)));

            ims.push_back(im_roi);
        }
    }

    void MTcnnDetector::bbreg(vector<vector<float> > &boxes)
    {
        for(int i=0; i<boxes.size(); ++i)
        {
            float w = boxes[i][2] - boxes[i][0];
            float h = boxes[i][3] - boxes[i][1];
            boxes[i][0] += w*boxes[i][5];
            boxes[i][1] += h*boxes[i][6];
            boxes[i][2] += w*boxes[i][7];
            boxes[i][3] += h*boxes[i][8];
            for(int j=5; j<boxes[i].size(); ++j)
            {
                boxes[i].erase(boxes[i].begin()+j);
                j--;
            }
        }
    }

    void MTcnnDetector::bbreg(vector<vector<float> > &boxes, const caffe::Blob<float> *prob, const caffe::Blob<float> *reg, const float threshold)
    {
        assert(prob->num()==boxes.size());
        const float *scores = prob->cpu_data();
        const float *boxreg = reg->cpu_data();
        int offset = 0;
        int temp_index = 0;

        for(int i=0; i<prob->num(); ++i)
        {
            if(scores[i*2+1]<=thresholds_[1])
            {
                boxes.erase(boxes.begin()+temp_index);
                continue;
            }
            boxes[temp_index][4] = scores[i*2+1];
            for(int j=0; j<4; ++j)
                boxes[temp_index].push_back(boxreg[i*4+j]);
            temp_index ++;
        }
        nms(boxes,outer_nms_);

        for(int i=0; i<boxes.size(); ++i)
        {
            float w = boxes[i][2] - boxes[i][0] + 1;
            float h = boxes[i][3] - boxes[i][1] + 1;
            boxes[i][0] += w*boxes[i][5];
            boxes[i][1] += h*boxes[i][6];
            boxes[i][2] += w*boxes[i][7];
            boxes[i][3] += h*boxes[i][8];
            for(int j=5; j<boxes[i].size(); ++j)
            {
                boxes[i].erase(boxes[i].begin()+j);
                j--;
            }
        }
    }

    void MTcnnDetector::rNet(const vector<cv::Mat> &ims, vector<vector<float> > &boxes)
    {
        wrapInputLayer(ims,1);
        std::shared_ptr<Net<float> > rnet = nets_[1];
        pair<string,string> out_blobname = out_blobnames_[1];

        rnet->Forward();
        Blob<float>* prob = rnet->blob_by_name(out_blobname.first).get();
        Blob<float>* reg = rnet->blob_by_name(out_blobname.second).get();

        bbreg(boxes,prob,reg,thresholds_[1]);
    }

    void MTcnnDetector::oNet(const vector<cv::Mat> &ims, vector<vector<float> > &boxes)
    {
        wrapInputLayer(ims,2);
        std::shared_ptr<Net<float> > onet = nets_[2];
        pair<string,string> out_blobname = out_blobnames_[2];

        onet->Forward();
        Blob<float>* prob = onet->blob_by_name(out_blobname.first).get();
        Blob<float>* reg = onet->blob_by_name(out_blobname.second).get();

        bbreg(boxes,prob,reg,thresholds_[2]);
        nms(boxes,outer_nms_);
    }

    void MTcnnDetector::detect(const cv::Mat &im, vector<ObjZoneDetect::Object> &objs, const float confidence_threshold)
    {
        cv::Mat im_temp = im.clone();
        cv::Mat im_float = imConvert(im);

        vector<Mat> ims;
        vector<float> scales;
        vector<vector<float> >dst_boxes;
        vector<Mat> im_rois;

        imResize(im_float,ims,scales);
        vector<vector<float> > boxes;
        pNet(ims,scales,boxes);
        bbreg(boxes);

        rerac(boxes);
        vector<vector<float> > boxes_bak = boxes;
        pad(boxes,im,dst_boxes);
        generateRois(im_float,boxes,dst_boxes,im_rois);
        boxes = boxes_bak;
        rNet(im_rois,boxes);

        rerac(boxes);
        printBBox(boxes);
        boxes_bak = boxes;
        pad(boxes,im,dst_boxes);
        printBBox(dst_boxes,boxes);
        generateRois(im_float,boxes,dst_boxes,im_rois);
        boxes = boxes_bak;
        oNet(im_rois,boxes);

        objs.clear();
        for(int i=0; i<boxes.size(); ++i)
        {
            if(boxes[i][4]<confidence_threshold)
                continue;
            ObjZoneDetect::Object obj;
            obj.zone.x = max(0,int(boxes[i][0]));
            obj.zone.y = max(0,int(boxes[i][1]));
            obj.zone.width = min(im.cols,int(boxes[i][2])) - obj.zone.x;
            obj.zone.height = min(im.rows,int(boxes[i][3])) - obj.zone.y;
            obj.cls = 0;
            obj.score = boxes[i][4];
            objs.push_back(obj);
        }
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

