#include<boost/python.hpp>
#include <numpy/arrayobject.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <IObjZoneDetect.h>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace boost::python;
using namespace ObjZoneDetect;
using namespace cv;

class ObjZoneDetector
{
public:
    ObjZoneDetector(const string &cfg_file,const string& weights_file ,const int type=0, const int gpu_id=0)
    {
        switch(type)
        {
            case 0:
            case 1:
            case 2:
                detector = CreateObjZoneYoloV3Detector(cfg_file,weights_file,gpu_id);
            default:
                break;
        }
    }

    vector<Object> detect(boost::python::object &data_obj, int w, int h,const float confidence_threshold)
    {
        PyArrayObject* data_arr = reinterpret_cast<PyArrayObject*>(data_obj.ptr());
        unsigned char* data = static_cast<unsigned char*>(PyArray_DATA(data_arr));

        Mat im(h,w,CV_8UC3,data);
        detector->detect(im,r,confidence_threshold);

        return r;
    }

private:
    ObjZoneDetect::IObjZoneDetect *detector;
    vector<Object> r;
};


BOOST_PYTHON_MODULE(_IObjZoneDetect)
{
    class_<cv::Rect>("zone",no_init)
            .add_property("x",&cv::Rect::x)
            .add_property("y",&cv::Rect::y)
            .add_property("w",&cv::Rect::width)
            .add_property("h",&cv::Rect::height);

    class_<Object>("PlateInfo",no_init)
            .add_property("zone",&Object::zone)
            .add_property("score",&Object::score)
            .add_property("cls",&Object::cls);

    class_<vector<ObjZoneDetect::Object> >("PlateInfos",no_init)
            .def(vector_indexing_suite<vector<ObjZoneDetect::Object> >())
            .def("size",&vector<ObjZoneDetect::Object>::size);

    class_<ObjZoneDetector>("ObjZoneDetector",init<const string&,const string&, const int, const int>())
            .def("detect",&ObjZoneDetector::detect);
}