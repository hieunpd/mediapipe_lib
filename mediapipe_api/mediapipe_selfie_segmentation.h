#ifndef MEDIAPIPE_SELFIE_SEGMENTATION
#define MEDIAPIPE_SELFIE_SEGMENTATION

#include <iostream>
#include <string>
#include <cstdlib>
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"


constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";

class MediaPipeSelfieSegmentation
{
private:

    bool isDetected;

    mediapipe::CalculatorGraph graph;

    std::unique_ptr<mediapipe::OutputStreamPoller> outputVideoPoller;
    // std::unique_ptr<mediapipe::OutputStreamPoller> outputLandmarksPoller;
    std::unique_ptr<mediapipe::OutputStreamPoller> outputHandCountPoller;

    // ::mediapipe::Packet handCountPacket;
    // ::mediapipe::Packet landmarksPacket;
    ::mediapipe::Packet imagePacket;


    std::string calculator_graph_config_contents;
  
    mediapipe::CalculatorGraphConfig config;
    
    size_t frame_timestamp_us;

protected:
    absl::Status Setup(std::string calculator_graph_config_file);

    absl::Status Shutdown();

public:
    MediaPipeSelfieSegmentation(bool &isDetected, std::string calculator_graph_config_file = "mediapipe/graphs/selfie_segmentation/selfie_segmentation_cpu.pbtxt");

    virtual ~MediaPipeSelfieSegmentation();
    
    absl::Status RunMPPGraph(cv::Mat &camera_frame_raw);
};

#endif // MEDIAPIPE_HANDS_DETECTOR