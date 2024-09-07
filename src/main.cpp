#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "AppProp.hpp"

int main(int argc, char* argv[]) {
    if(argc!=2 && argc!=3){
        std::cout<<"usage: AppProp.exe image_filename";
        return 0;
    }

    std::string filename = argv[1];
    cv::Mat img = cv::imread(filename, 1);
    if(img.empty()){
        std::cout<<"img invalid";
        return 0;
    }

    bool extra = 0;
    if(argc==3 && std::string(argv[2])=="-extra") extra = 1;

    APGUI::gui_main(img, extra);

    return 0;
}
