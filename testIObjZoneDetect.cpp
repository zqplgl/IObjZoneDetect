//
// Created by zqp on 18-8-1.

#include<opencv2/opencv.hpp>
#include<iomanip>
#include <iosfwd>
#include<iostream>
#include <time.h>
#include "IObjZoneDetect.h"

using namespace std;
using namespace cv;
using namespace ObjZoneDetect;
void addRectangle(cv::Mat &img, const vector<Object> &objs)
{
	for (int i = 0; i < objs.size(); i++)
	{
		const Rect &zone = objs[i].zone;
		rectangle(img, cv::Point(zone.x, zone.y),  cv::Point(zone.x + zone.width, zone.y + zone.height), Scalar(0, 0, 255), 2);
		stringstream strstream;

		strstream <<setiosflags(ios::fixed);
		strstream << objs[i].cls << ":" <<setprecision(1) << objs[i].score;
		string label;
		strstream >> label;
		putText(img, label, Point(zone.x, zone.y), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	}
}

void getImPath(string& picDir,vector<string>&imPath)
{
    string cmd = "find "+picDir+" -name *.jpg";
    FILE *fp = popen(cmd.c_str(),"r");
    char buffer[512];
    while(1)
    {
        fgets(buffer,sizeof(buffer),fp);
        if(feof(fp))
            break;
        buffer[strlen(buffer)-1] = 0;
        imPath.push_back(string(buffer));
    }
}


int main(int argc , char** argv)
{

	string cfg_file ="/home/zqp/install_lib/models/yolov3/yolov3.cfg";
    string weights_file = "/home/zqp/install_lib/models/yolov3/yolov3.weights";

    string picDir = "/home/zqp/mygithub/darknet/data/";

    int gpu_id = 0;
    IObjZoneDetect *detector = CreateObjZoneYoloV3Detector(cfg_file,weights_file,gpu_id);

    vector<string> imPath;
    getImPath(picDir,imPath);

    string file;
    vector<Object> objs;

    clock_t start,end;
    int count  = 0;

    int index = 0;
    if(!imPath.size())
        return 0;

    while(1)
    {

        if(index>=imPath.size())
            index = 0;

        cout<<imPath[index]<<endl;
        file = imPath[index];
        cv::Mat im = cv::imread(file);
        cv::resize(im,im,Size(),0.25,0.25);
        cout<<"im.size: "<<im.size()<<endl;
        string picName = file.substr(file.find_last_of("/")+1);
        picName = picName.substr(0,picName.size()-4);

        start = clock();
        detector->Detect(im,objs,0.5);
        end = clock();
        cout<<picName<<"**********detect cost time: "<<(double(end-start)/CLOCKS_PER_SEC)*1000<<" ms"<<endl;
        cout<<"objs: "<<objs.size()<<endl;
        addRectangle(im,objs);
        index += 1;
        imshow("im",im);
        waitKey(0);
  }
    //DestroyObjZoneFasterDetector(detector);
}
