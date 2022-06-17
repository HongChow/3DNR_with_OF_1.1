//
// Created by hong on 22-5-14.
//

#ifndef INC_3DNR_OF_WARP_WITH_FLOW_H
#define INC_3DNR_OF_WARP_WITH_FLOW_H
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <cmath>
using namespace std;
using namespace cv;
__inline int CLIP(int indata,int min, int max)
{
    if (indata<min)
        return min;
    else if (indata>max)
        return max;
    else
        return indata;
}
cv::Mat WarpFrame(cv::Mat rgbImage,cv::Mat flow,int width, int height);
cv::Mat calculate_divergence(cv::Mat flow,int width, int height);
cv::Mat distance_mask(cv::Mat rgbImage,cv::Mat rgbImageW_seq,cv::Mat divergence,int width,int height,float mask_sigma);
cv::Mat CheckLeftRightFlow(cv::Mat flow,cv::Mat flow_inv,int width,int height,float lrflowTh);
cv::Mat InterSect(cv::Mat maskdist,cv::Mat masklr,int width,int height);
#endif //INC_3DNR_OF_WARP_WITH_FLOW_H

