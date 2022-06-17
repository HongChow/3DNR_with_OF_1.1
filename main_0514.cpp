/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  this is the 1.0 version of 3DNR based with optical flow estimation
 *
 *        Version:  1.0
 *        Created:  2022年05月11日 15时59分39秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  ZhouHong＠cistadesgin.com,chinamos@sjtu.edu.cn 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

const char * inpath = "deadline_cif.yuv";
const char* outPath = "out.yuv";

cv::Mat prevgray, gray, rgb, frame;
cv::Mat flow, flow_uv[2];
cv::Mat mag, ang;
cv::Mat hsv_split[3], hsv;
cv::Ptr<DenseOpticalFlow> flow_algorithm = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
cv::Mat yuvImage_pre, yuvImage_cur,rgbImage_pre,rgbImage_cur,y_pre,y_cur;
int main()
{
    std::cout<<"this is a test"<<std::endl; 
    cv::Mat img = imread("noisy_image.bmp");
    if(img.empty()){
        std::cout<<"no valid image"<<std::endl;
        return -1;
    }
    ifstream in(inpath, ios::binary);
    in.seekg(0, ios::end);
    int size = in.tellg();
    in.seekg(0, ios::beg);
    std::cout<<"size = "<<size<<std::endl;
    char* inBuffer = new char[size];
    in.read(inBuffer, size);
    in.close();
    char* outBuffer = new char[size];
    int width = 352, height = 288, inFrameSize = width * height * 3 / 2,yFrameSize = width * height;
    yuvImage_pre.create(height*3/2,width,CV_8UC1);
    y_pre.create(height,width,CV_8U);
    y_cur.create(height,width,CV_8U);
    rgbImage_pre.create(height,width,CV_8UC3);
    int frameNum = size / inFrameSize;

    for (int i = 0; i < frameNum; ++i){
        //for (int j = 0; j < inFrameSize; ++ j){
	//    outBuffer[i * inFrameSize + j] = j < width * height ? inBuffer[i * inFrameSize + j] : 128;
        //}
        if (i<3)
        {
            std::cout<<"pass"<<std::endl;
        }
        else 
        {
            //memcpy(yuvImage_pre.data,inBuffer,inFrameSize);
            //memcpy(yuvImage_pre.data,inBuffer,inFrameSize*sizeof(char));
            memcpy(yuvImage_pre.data,inBuffer+(i-1)*inFrameSize,inFrameSize);
            //memcpy(yuvImage_cur.data,inBuffer+(i)*inFrameSize,inFrameSize);
            //memcpy(yuvImage_cur.data,inBuffer+(i)*inFrameSize,inFrameSize);
            memcpy(y_pre.data,inBuffer+(i-1)*inFrameSize,yFrameSize);
            memcpy(y_cur.data,inBuffer+(i)*inFrameSize,yFrameSize);
            flow_algorithm->calc(y_pre, y_cur, flow);
            cv::split(flow, flow_uv);
            multiply(flow_uv[1], -1, flow_uv[1]);
            cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);
            normalize(mag, mag, 0, 1, NORM_MINMAX);
            hsv_split[0] = ang;
            hsv_split[1] = mag;
            hsv_split[2] = Mat::ones(ang.size(), ang.type());
            merge(hsv_split, 3, hsv);
            cvtColor(hsv, rgb, COLOR_HSV2BGR);
            cv::Mat rgbU;
            rgb.convertTo(rgbU, CV_8UC3,  255, 0);
            cv::imshow("DISOpticalFlow", rgb);

            cv::cvtColor(yuvImage_pre,rgbImage_pre,COLOR_YUV2RGB_YV12);
            cv::imshow("pre image",rgbImage_pre);
            cv::waitKey(0);
        }
        /*
            algorithm->calc(prevgray, gray, flow);
            split(flow, flow_uv);
            multiply(flow_uv[1], -1, flow_uv[1]);
            cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);
            normalize(mag, mag, 0, 1, NORM_MINMAX);
            hsv_split[0] = ang;
            hsv_split[1] = mag;
            hsv_split[2] = Mat::ones(ang.size(), ang.type());
            merge(hsv_split, 3, hsv);
            cvtColor(hsv, rgb, COLOR_HSV2BGR);
            cv::Mat rgbU;
            rgb.convertTo(rgbU, CV_8UC3,  255, 0);
            cv::imshow("DISOpticalFlow", rgbU);
            */
    }

    //ofstream out(outPath,ios::binary);
    //out.write(outBuffer, size);
    //out.close();


    cv::imshow("test image",img);
    cv::waitKey(0);
    return 0;
}


