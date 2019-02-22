#include "YOLOManager.h"

YOLOManager::YOLOManager(std::string classesFile, std::string _modelConfiguration, std::string _modelWeights,
    float _confThreshold, float _nmsThreshold, int _inpWidth, int _inpHeight, int _imgWidth, int _imgHeight)
{
    modelConfiguration = _modelConfiguration;
    modelWeights = _modelWeights;

    // Load names of classes
    std::ifstream classNamesFile(classesFile.c_str());
    if (classNamesFile.is_open())
    {
        std::string className = "";
        while (std::getline(classNamesFile, className))
            classes.push_back(className);
    }
    else{
        std::cout<<"can not open classNamesFile"<<std::endl;
    }
     // Give the configuration and weight files for the model
    // Load the network
    net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    std::cout<<"Read Darknet..."<<std::endl;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    confThreshold = _confThreshold;
    nmsThreshold = _nmsThreshold;
    inpWidth = _inpWidth;
    inpHeight = _inpHeight;
    imgWidth = _imgWidth;
    imgHeight = _imgHeight;    
}

void
YOLOManager::DetectObjects(const cv::Mat &_frame)
{
    frame = _frame.clone(); 
    cv::dnn::blobFromImage(frame, blob, 1.0/255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0,0,0), true, false);
    //Sets the input to the network
    net.setInput(blob);

    // Create a 4D blob from a frame.
    // Runs the forward pass to get output of the output layers
    std::vector<cv::Mat> outs;
    net.forward(outs, getOutputsNames());
    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);
}
    
std::vector<cv::String>
YOLOManager::getOutputsNames()
{
    static std::vector<cv::String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
         
        //get the names of all the layers in the network
        std::vector<cv::String> layersNames = net.getLayerNames();
         
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void 
YOLOManager::postprocess(cv::Mat& frame, std::vector<cv::Mat>& outs)
{
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;

        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            
            //cv::Mat scores = outs[i].row(j);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > confThreshold)
//            if (scores.at<double>(0,0) > confThreshold)
 //           if (true)
            {
//                cout << scores.at<double>(0,0) << endl;
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                 
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                //confidences.push_back(float(scores.at<double>(0,0)));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences

    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
}

cv::Mat 
YOLOManager::drawPred()
{
    cv::Mat out_frame; 
    out_frame = frame.clone();
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        _drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, out_frame );
    }
    return out_frame;
}

// Draw the predicted bounding box
void 
YOLOManager::_drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &out_frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(out_frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255),20);
     
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    else
    {
        std::cout<<"classes is empty..."<<std::endl;
    }
     
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 3.0, 1.0, &baseLine);
    top = std::max(top, labelSize.height);
    cv::putText(out_frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255,255,255), 8);
}

double
YOLOManager::getLayersTime()
{
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    return t;
}

void
YOLOManager::refresh()
{
    classIds.clear();
    confidences.clear();
    boxes.clear();
    indices.clear();
}

