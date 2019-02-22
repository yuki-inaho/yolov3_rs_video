#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <dirent.h>
#include <unistd.h>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h> // Include RealSense Cross Platform API

#include <opencv2/videoio.hpp>

#include "YOLOManager.h"
#include "utils.hpp"

using namespace std; 
// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs);
 
// Get the names of the output layers
std::vector<cv::String> getOutputsNames(const cv::dnn::Net& net);
 
// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.5;  // Non-maximum suppression threshold
int inpWidth = 608;        // Width of network's input image
int inpHeight = 608;       // Height of network's input image
 
int main(int argc, char** argv)
{
	cv::VideoWriter writer("../data/aspara_0001.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'S'), 2.0, cv::Size(640, 360));
	//動画ファイルがちゃんと作れたかの判定。
	if (!writer.isOpened()){ return -1; }

    rs2::pipeline pipe;
    rs2::config cfg;

    std::string loadBagFileName = "../data/bag_0001.bag";
    cfg.enable_device_from_file(loadBagFileName);
    pipe.start(cfg);

    std::string classesFile = "../data/aspara.names"; 
    std::string modelConfiguration = "../data/aspara.cfg";
    std::string modelWeights = "../data/aspara.weights";
    std::cout <<"Processing..."<<std::endl;

    cv::Mat _frame, frame,frame_resized;
    _frame = cv::imread("../img/color_0000.jpg");
    YOLOManager yolo_mng (classesFile, modelConfiguration, modelWeights, 
                            confThreshold, nmsThreshold, inpWidth, inpHeight, frame.rows, frame.cols);

    int key;
    bool out_flag = true;
    while (1)
    {
        rs2::frameset frames = pipe.wait_for_frames();
        // RGBカメラの撮影結果
        rs2::video_frame rgb = frames.get_color_frame();
        cv::Mat frame(cv::Size(rgb.get_width(), rgb.get_height()), CV_8UC3, (void*)rgb.get_data());
        cv::cvtColor(frame, frame, CV_RGB2BGR);
    
        // Stop the program if reached end of video
        if (frame.empty()) {
            if(cv::waitKey(0)==27)
            break;
        }
        //show frame
        cv::resize(frame, frame_resized,cv::Size(),0.5,0.5);    
        cv::imshow("frame",frame_resized);

        yolo_mng.DetectObjects(frame);
        // Put efficiency information. The function getPerfProfile returns the 
        // overall time for inference(t) and the timings for each of the layers(in layersTimes)
        
        double t = yolo_mng.getLayersTime();
        std::string label = cv::format("Inference time for a frame : %.2f ms", t);
        std::cout << "Inference time for a frame:" << t << "ms" <<std::endl;
        cv::putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        
        // Write the frame with the detection boxes
        cv::Mat detectedFrame, detectedFrame_resized;
        detectedFrame = yolo_mng.drawPred();
        detectedFrame.convertTo(detectedFrame, CV_8U);
        cv::resize(detectedFrame, detectedFrame_resized,cv::Size(),0.5,0.5);    

        //show detectedFrame
        cv::imshow("detectedFrame",detectedFrame_resized);
        //save result
        //cv::imwrite("../img/res.jpg", detectedFrame);
        /*
        if(out_flag){
            string out_name = "../img/output/res_" + num_str + ".jpg";
            cv::imwrite(out_name, detectedFrame);
            out_flag = false;
        }
        */
        writer << detectedFrame_resized;
        yolo_mng.refresh();
        key = cv::waitKey(50);
        if(27 == key)
        {
            break;
        }
//        if(key == 'n')
    }

    std::cout<<"Esc..."<<std::endl;
    return 0;
}
 
