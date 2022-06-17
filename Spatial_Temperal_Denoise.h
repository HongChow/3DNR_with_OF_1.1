//
// Created by hong on 22-5-24.
//

#ifndef INC_3DNR_OF_SPATIAL_TEMPERAL_DENOISE_H
#define INC_3DNR_OF_SPATIAL_TEMPERAL_DENOISE_H
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <cmath>
#include <chrono>
struct denoise_para{
   int width;
   int height;
   int channels;
   int radius_search;
   int radius_block;
   int radius_Tem;// ---frame nums used = 2*radius_Tem
   float  oflat;
   float osigma;
   float ofpca;
};
struct diffpatches {
    int i, j;
    double diff;
};
struct patch_selected{
    int i,j; // ｉ,j 坐标
    int seq_idx;// warp id
    double diff;
    cv::Mat block_img;// block image 
    std::vector<cv::Mat> block_img_vects; // block image of each channels
};
void Spatial_Temperal_Denoise(cv::Mat rgbImageW_seq[],cv::Mat current_rgbImage,float ** R_Warped[],float **G_Warped[],float ** B_Warped[],denoise_para param,float ** Weights_Mask,std::vector<cv::Mat>& OutPutImages);
void NormalizeOutput(cv::Mat current_rgbImage, float **Weights_Mask,std::vector<cv::Mat> OutPutImages,cv::Mat &OutPutFrame,int height,int width);
#endif //INC_3DNR_OF_SPATIAL_TEMPERAL_DENOISE_H
