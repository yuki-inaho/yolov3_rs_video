#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <iostream>
#include <fstream>

class YOLOManager{
    public:
    
    YOLOManager(std::string classesFile, std::string _modelConfiguration, std::string _modelWeights, 
        float _confThreshold, float _nmsThreshold, int _inpWidth, int _inpHeight, int _imgWidth, int _imgHeight);
    void DetectObjects(const cv::Mat &frame);
    double getLayersTime();
    cv::Mat drawPred();
    void refresh();

    private:
    std::vector<cv::String> getOutputsNames();    
    void postprocess(cv::Mat &frame, std::vector<cv::Mat>& outs);
    void _drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &out_frame);
    


    cv::String modelConfiguration;
    cv::String modelWeights;
    std::vector<std::string> classes;
    cv::dnn::Net net;
    cv::Mat blob;
    cv::Mat frame;

    float confThreshold;
    float nmsThreshold;
    int inpWidth;
    int inpHeight;
    int imgWidth;
    int imgHeight;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> indices;
};