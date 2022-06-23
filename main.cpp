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
 *         Author:  ZhouHong＠cistadesgin.com; chinamos@sjtu.edu.cn
 *   Organization:  
 *
 * =====================================================================================
 */

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <omp.h>
#include  "display_flow.h"
#include  "optical_flow_related.h"
#include "Spatial_Temperal_Denoise.h"
#include "CNR_Spatial.h"
using namespace std;
using namespace cv;
//#define CISTA
#ifdef CISTA
    //const char * inpath = "cj1_part.yuv";
    const char * inpath = "cj2_ori.yuv";
    int width = 1920, height = 1080;
#else
const char * inpath = "hall_monitor_cif.yuv";
int width = 352, height = 288;
#endif
#define PreCNR
#define DEBUG_CNR
int uv_height = height>>1;
int uv_width = width>>1;
//const char * inpath = "1.yuv";
//const char * inpath = "hall_monitor_cif.yuv";
const char* outPath = "out.yuv";

const int Nums = 7;
const int middle = Nums/2;
cv::Mat prevgray, gray, rgb, frame;
cv::Mat flow, flow_uv[2];
cv::Mat mag, ang;
cv::Mat hsv_split[3], hsv;
cv::Ptr<DenseOpticalFlow> flow_algorithm = cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM);
cv::Mat yuvImage_pre, yuvImage_cur,rgbImage_pre,rgbImage_cur,y_pre,y_cur;
cv::Mat yuvImage_seq[Nums],rgbImage_seq[Nums],y_seq[Nums],rgbImageW_seq[Nums-1];
cv::Mat rgbImageCNR_seq[Nums];
cv::Mat rgbImage_current,rgbImageCNR_current,rgbImage_Denoised;
cv::Mat flow_seq[Nums-1],flow_inv_seq[Nums-1],flow_rgb[Nums-1],divergence_seq[Nums-1],maskdist_seq[Nums -1],outmask_seq[Nums-1];
std::vector<cv::Mat> OutPutImages(3); // for denoised frames of each channels//
std::vector<cv::Mat> CurrentImagesVectors(3);
cv::Mat OutPutFrame; // the final denoised image //

cv::Mat temp;
int main() {
    std::cout << "this is a OMP test" << std::endl;
//#pragma omp parallel
//    {
//        int ID = omp_get_thread_num();
//        std::cout<<"hello("<<ID<<")";
//        std::cout<<"world("<<ID<<")"<<std::endl;
//    }
    //cv::Mat img = imread("of1.png");
    //std::cout<<img.dims<<" -             -- -        -"<<std::endl;
    //std::cout<<img.size<<" -             -- -        -"<<std::endl;
    //std::cout<<img.channels()<<" -             -- -        -"<<std::endl;
    float sigma = 10;
    float lrflowTh = 15.0;
    float mask_sigma = 1*sigma;
    ifstream in(inpath, ios::binary);
    in.seekg(0, ios::end);
    int size = in.tellg();
    in.seekg(0, ios::beg);
    std::cout << "size = " << size << std::endl;
    char *inBuffer = new char[size];
    in.read(inBuffer, size);
    in.close();
    char *outBuffer = new char[size];
    //int width = 352, height = 288, inFrameSize = width * height * 3 / 2, yFrameSize = width * height;
    int inFrameSize = width * height * 3 / 2, yFrameSize = width * height;
    float **Weights_Mask = new float *[height];
    for (int i=0; i<height; i++){
        Weights_Mask[i] = new float [width];
    }
    for (int i=0;i<height; i++)
        for(int j=0; j< width; j++)
            Weights_Mask[i][j] = 0.0f;
    yuvImage_pre.create(height * 3 / 2, width, CV_8UC1);
    y_pre.create(height, width, CV_8U);
    y_cur.create(height, width, CV_8U);
    rgbImage_pre.create(height, width, CV_8UC3);
    rgbImage_current.create(height,width,CV_8UC3);
    OutPutFrame.create(height,width,CV_8UC3);
    for (int i =0; i<3; i++){
        OutPutImages[i].create(height,width,CV_32FC1);
    }
    //std::cout<<rgbImage_pre.dims<<" -             -- -        -"<<std::endl;
    //===== Param init for 3D Denoise ===== //
    denoise_para para3D;
    para3D.width = width;para3D.height = height; para3D.channels = 3; para3D.radius_Tem = middle;
    para3D.radius_block = 2; para3D.radius_search = 6;
    para3D.oflat = 0.85f;para3D.osigma=5.0f;para3D.ofpca = 1.8f;
    //===== Param init for Spatial CNR ===== //
    CNR_Spatial_para cnr_para;
    cnr_para.iWidth = width>>1;cnr_para.iHeight = height>>1; cnr_para.ratio_x = 2; cnr_para.ratio_y=2; cnr_para.r_patch_v = 4;cnr_para.r_patch_h=8;cnr_para.esp=32;

    // --------- allocate memory for adjacent frames -------- //
    for (int i = 0; i < Nums; i++) {
        yuvImage_seq[i].create(height * 3 / 2, width, CV_8U);
        y_seq[i].create(height, width, CV_8U);
        rgbImage_seq[i].create(height, width, CV_8UC3);
        rgbImageCNR_seq[i].create(height, width, CV_8UC3);
        //rgbImageW_seq[i].create(height, width, CV_8UC3);
        //std::cout << rgbImage_seq[i].channels() << " -             -- -        -" << std::endl;
    }
    for (int i = 0; i < Nums-1; i++) {
        divergence_seq[i].create(height,width,CV_32FC1);
        maskdist_seq[i].create(height,width,CV_32FC1);
        outmask_seq[i].create(height,width,CV_32FC1);
    }
    for (int i = 0; i < Nums - 1; i++) {
        rgbImageW_seq[i].create(height, width, CV_8UC3);
    }
    int frameNum = size / inFrameSize;

    for (int i = 0; i < frameNum; ++i) {
        //std::cout << "------------ current frame is ------------------ " << i << "  --------------\n" << std::endl;
        //for (int j = 0; j < inFrameSize; ++ j){
        //    outBuffer[i * inFrameSize + j] = j < width * height ? inBuffer[i * inFrameSize + j] : 128;
        //}
        //    ----------------------  Update Data Buffer --------------------------- //
        if (i < 3) {
            std::cout << "pass" << std::endl;
        }
        else if (i == 3) {
            // -------- initialization of frames ---------- //
            for (int ii = 0; ii < Nums; ++ii) {
                memcpy(yuvImage_seq[ii].data, inBuffer + (ii) * inFrameSize, inFrameSize);
                memcpy(y_seq[ii].data, inBuffer + (ii) * inFrameSize, yFrameSize);
                //std::cout<<rgbImage_seq[i].channels()<<" -             -- -        -"<<std::endl;
                cv::cvtColor(yuvImage_seq[ii], rgbImage_seq[ii], COLOR_YUV2RGB_YV12);
                // --- fetch yuv data and do color noise reduction --- //
                //std::cout<<rgbImage_seq[i].channels()<<" -             -- -        -"<<std::endl;
            }
            rgbImage_seq[3].copyTo(rgbImage_current);
// －－－－－－－－－－－－－－－　Convert the YUV420 Data to 2-D Array for Pre Color Noise Reduction －－－－－－－－－－－－－//
            float ** Y_data = new float * [uv_height];
            float ** U_data = new float * [uv_height];
            float ** V_data = new float * [uv_height];
            float ** Uout_data = new float * [uv_height];
            float ** Vout_data = new float * [uv_height];

            for (int row = 0; row<uv_height; row++){
                Y_data[row] = new float [uv_width];
                U_data[row] = new float [uv_width];
                V_data[row] = new float [uv_width];
                Uout_data[row] = new float [uv_width];
                Vout_data[row] = new float [uv_width];
            }
            cv::Mat y_seq_ds;
            y_seq_ds.create(uv_height,uv_width,CV_8UC1);
            cv::Size ds_size(uv_width,uv_height);
            for(int frame_id = 0; frame_id<Nums; frame_id++){
                cv::resize(y_seq[frame_id],y_seq_ds,ds_size);
                for(int row = 0; row<uv_height; row++)
                    for(int col = 0; col <uv_width; col++) {
                        Y_data[row][col] = (float)y_seq_ds.at<uchar>(row, col);
                    }
                // －－－－－－－－－－－－－－－－－－－－－－－－－－－ 从yuvImage_seq中获取ＵＶ的数据　－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－//
                for(int row_uv = 0; row_uv<uv_height; row_uv++) {
                    int row_buffer = row_uv >> 1;
                    int row_half_flag = row_uv % 2;
                    for (int col_uv = 0; col_uv < uv_width; col_uv++) {
                        int col_buffer = row_half_flag == 0 ? col_uv : col_uv + uv_width;
                        U_data[row_uv][col_uv] = (float) yuvImage_seq[0].at<uchar>(1080 + row_buffer, col_buffer);
                        V_data[row_uv][col_uv] = (float) yuvImage_seq[0].at<uchar>(1350 + row_buffer, col_buffer);
                    }
                }
                // －－－－－－－－－－－－－－－－－－－　Do Pre Color Noise Reduction －－－－－－－－－－－－－－－－－－－－　//
                CNR_Spatial_Top(cnr_para,       // Filtering parameter
                                Y_data,     // guided image --- padded
                                U_data,     // noisy  image --- Padded
                                V_data,     // noisy  image --- Padded
                                Uout_data,
                                Vout_data);
                // －－－－－－－－－－－－　Save The Pre Color Noise Reduction to YUV Mat Data －－－－－－－－－－－－－－　//
                cv::Mat yuvImage_preCNR;
                yuvImage_seq[frame_id].copyTo(yuvImage_preCNR);
                yuvImage_preCNR.create(height*3>>1,width,CV_8U);
                for(int row_uv = 0; row_uv<uv_height; row_uv++){
                    int row_buffer = row_uv>>1;
                    int row_half_flag = row_uv%2;
                    for(int col_uv = 0; col_uv<uv_width; col_uv++){
                        int col_buffer = row_half_flag == 0 ? col_uv : col_uv + uv_width;
                        yuvImage_preCNR.at<uchar>(1080+row_buffer, col_buffer) = Uout_data[row_uv][col_uv];
                        yuvImage_preCNR.at<uchar>(1350 + row_buffer, col_buffer) = Vout_data[row_uv][col_uv];
                    }
                }
                cv::Mat rgbImage_CNR;
                cv::cvtColor(yuvImage_preCNR, rgbImage_CNR, COLOR_YUV2RGB_YV12);
                std::string rgbImage_CNR_name = "rgbImage_CNR_vRadius_"+std::to_string(cnr_para.r_patch_v)+"_eps_"+std::to_string((int)cnr_para.esp)+"_"+std::to_string(frame_id)+".png";
                std::string rgbImagename = "rgbImage_"+std::to_string(frame_id)+".png";
                cv::imwrite(rgbImage_CNR_name,rgbImage_CNR);
                cv::imwrite(rgbImagename,rgbImage_seq[frame_id]);
                rgbImage_CNR.copyTo(rgbImageCNR_seq[frame_id]);
            }
            rgbImageCNR_seq[3].copyTo(rgbImageCNR_current);
            //exit(0);

        } else { // i > 3 //
            // ------------- update the frame buffer and current frame -------------- //
            for (int ii = 1; ii < Nums; ++ii) {
                yuvImage_seq[ii].copyTo(yuvImage_seq[ii - 1]);
                y_seq[ii].copyTo(y_seq[ii - 1]);
                rgbImage_seq[ii].copyTo(rgbImage_seq[ii - 1]);
                //std::cout<<rgbImage_seq[i].channels()<<" -             -- -        -"<<std::endl;
            }
            memcpy(yuvImage_seq[Nums - 1].data, inBuffer + (i) * inFrameSize, inFrameSize);
            memcpy(y_seq[Nums - 1].data, inBuffer + (i) * inFrameSize, yFrameSize);
            cv::cvtColor(yuvImage_seq[Nums - 1], rgbImage_seq[Nums - 1], COLOR_YUV2RGB_YV12);
            rgbImage_seq[3].copyTo(rgbImage_current);
            //cv::imshow("rgbImage_seq[3]",rgbImage_seq[3]);
            //std::cout<<rgbImage_seq[Nums - 1].channels()<<" -             -- -        -"<<std::endl;
            // -------------- Calculation of optical flow for each adjacent frame ------------- //
        }
        // ------------------- Calculation of optical flow and warp frame adjacent ---------------- //
        if (i < 3) {
            std::cout << "pass" << std::endl;
        } else {
            for (int ii = -3; ii <= 3; ii++) {
                if (ii != 0) {
                    int flow_idx = ii + middle > 3 ? ii + middle - 1 : ii + middle;
                    flow_algorithm->calc(y_seq[middle], y_seq[middle + ii], flow_seq[flow_idx]);
                    flow_rgb[flow_idx] = flow_rgb_cal(flow_seq[flow_idx]);
                    std::string DISFlow_name = "DISFlow_" + std::to_string(ii);
                    cv::imwrite(DISFlow_name+".png",flow_rgb[flow_idx]);
                    //std::cout<<rgbImage_seq[middle+ii].channels()<<" -             -- -        -"<<std::endl;
#ifdef PreCNR
                    rgbImageW_seq[flow_idx] = WarpFrame(rgbImageCNR_seq[middle + ii], flow_seq[flow_idx], width, height);
#else
                    rgbImageW_seq[flow_idx] = WarpFrame(rgbImage_seq[middle + ii], flow_seq[flow_idx], width, height);
#endif
                    // ------------------ Calculation of OcclusionMask -------------------- //
                    // -------- step1 calculation divergence ----------- //
                    divergence_seq[flow_idx] = calculate_divergence(flow_seq[flow_idx],width,height);
                    // -------- step2 calculation difference between warped image and target image  ----------- //
                    maskdist_seq[flow_idx] = distance_mask(rgbImage_seq[middle+ii],rgbImageW_seq[flow_idx],divergence_seq[flow_idx],width,height,mask_sigma);
                    // -------- step3 calculation of inverse flow --------- //
                    flow_algorithm->calc(y_seq[middle + ii],y_seq[middle],  flow_inv_seq[flow_idx]);
                    //cv::Mat mask_temp;
                    //std::string outmask_name = "outmask_" + std::to_string(flow_idx);
                    outmask_seq[flow_idx] = InterSect(maskdist_seq[flow_idx],CheckLeftRightFlow(flow_seq[flow_idx],flow_inv_seq[flow_idx],width,height,lrflowTh),width,height);
                    //outmask_seq[flow_idx].convertTo(mask_temp, CV_8U, 255.0/(1.0-0.0));
                    //cv::imwrite(outmask_name+".png",outmask_seq[flow_idx]);
                    //std::cout<<"------------- log4.3 ----------------\n";
                    // ---------- 3D Denoise ---------- //

                    // ---------- Step1. Get 3D Blocks ----------- //
                }

                //else
                //    rgbImage_seq[middle].copyTo(rgbImageW_seq[middle]); ---这不对
            }
            //exit(0);
            if (i==3) {
                std::vector<std::vector<cv::Mat >> rgbImage_W_Vects(Nums-1);
                for (int t = 0; t < Nums-1; t++) {
                    cv::split(rgbImageW_seq[t], rgbImage_W_Vects[t]);
                }
                // ----------- Convert CV::Mat to ** pointer -------------- //
                float ** R_Warped[Nums-1];
                float ** G_Warped[Nums-1];
                float ** B_Warped[Nums-1];

                //std::vector<float **>R_Warped(Nums-1);
                //std::vector<float **>G_Warped(Nums-1);
                //std::vector<float **>B_Warped(Nums-1);
                // -----------Allocate Memory --------------- //
                for (int idx = 0; idx<Nums-1;idx++){
                    R_Warped[idx] = new float * [height];
                    G_Warped[idx] = new float * [height];
                    B_Warped[idx] = new float * [height];
                    for (int row = 0; row<height;row++){
                        R_Warped[idx][row] = new float [width];
                        G_Warped[idx][row] = new float [width];
                        B_Warped[idx][row] = new float [width];
                    }
                }
                // ------------- Init --------------- //
                for (int idx = 0; idx<Nums-1;idx++){
                    for (int row = 0; row<height;row++){
                        for (int col = 0; col<width;col++){
                            R_Warped[idx][row][col] =  (float)rgbImage_W_Vects[idx][0].at<uchar>(row,col);
                            G_Warped[idx][row][col] =  (float)rgbImage_W_Vects[idx][1].at<uchar>(row,col);
                            B_Warped[idx][row][col] =  (float)rgbImage_W_Vects[idx][2].at<uchar>(row,col);
                        }
                    }
                }

                //cv::imshow("rgbImage_current_ori",rgbImage_current);
                for (int ii = -3; ii <= 3; ii++) {
                    if (ii != 0) {
                        int flow_idx = ii + middle > 3 ? ii + middle - 1 : ii + middle;
                        std::string warp_name = "WrapYUV_" + std::to_string(ii);
                        cv::imwrite(warp_name+".png",rgbImageW_seq[flow_idx]);
                    }else{
                        std::string rgb_name = "WrapYUV_" + std::to_string(0);
                        cv::imwrite(rgb_name+".png", rgbImage_seq[middle]);
                    }
                }
#ifdef  PreCNR
                cv::split(rgbImage_current,CurrentImagesVectors);
                auto beforeCoreTime = std::chrono::steady_clock::now();
                Spatial_Temperal_Denoise(rgbImageW_seq, rgbImageCNR_current, R_Warped,G_Warped,B_Warped,para3D,Weights_Mask,OutPutImages);
                auto afterCoreTime = std::chrono::steady_clock::now();
                double duration_millsecondCore = std::chrono::duration<double, std::milli>(afterCoreTime - beforeCoreTime).count();
                std::cout <<"Spatial_Temperal_Denoise"<< duration_millsecondCore << "毫秒" << std::endl;
                // -------------Denoise Results Normalized --------------- //
                std::cout<<" ---------- running finished before Normalize -----------------"<<std::endl;
                NormalizeOutput(rgbImage_current,Weights_Mask,OutPutImages,OutPutFrame,height,width);
                std::cout<<" ---------- running finished before saving -----------------"<<std::endl;
                //cv::imwrite("/media/hong/62CC6F80CC6F4D7B/3DNR/3DNR_with_OF_1.1/build/DenoisedwoPreCNR.png",OutPutFrame);
                //cv::imwrite("/home/hong/3DNR/3DNR_with_OF_1.1/build/Denoised_Results/DenoisedwoPreCNR3.png",OutPutFrame);
                cv::imwrite("Denoised_cj2_PreCNR3.png",OutPutFrame);

                exit(0);
#else
                cv::split(rgbImage_current,CurrentImagesVectors);
                auto beforeCoreTime = std::chrono::steady_clock::now();
                Spatial_Temperal_Denoise(rgbImageW_seq, rgbImage_current, R_Warped,G_Warped,B_Warped,para3D,Weights_Mask,OutPutImages);
                auto afterCoreTime = std::chrono::steady_clock::now();
                double duration_millsecondCore = std::chrono::duration<double, std::milli>(afterCoreTime - beforeCoreTime).count();
                std::cout <<"Spatial_Temperal_Denoise"<< duration_millsecondCore << "毫秒" << std::endl;
                // -------------Denoise Results Normalized --------------- //
                std::cout<<" ---------- running finished before Normalize -----------------"<<std::endl;
                NormalizeOutput(rgbImage_current,Weights_Mask,OutPutImages,OutPutFrame,height,width);
                std::cout<<" ---------- running finished before saving -----------------"<<std::endl;
                //cv::imwrite("/media/hong/62CC6F80CC6F4D7B/3DNR/Implementation/3DNR_with_OF_1.1/build/DenoisedwoPreCNR.png",OutPutFrame);
#endif

                //cv::imwrite("/home/hong/3DNR/3DNR_with_OF_1.1/build/Denoised_Results/Denoised.png",OutPutFrame);

                //exit(0);
            }
            // ------------------- just for display -------------------- //
            for (int ii = -3; ii <= 3; ii++) {
                if (ii != 0) {
                    int flow_idx = ii + middle > 3 ? ii + middle - 1 : ii + middle;
                    std::string DISFlow_name = "DISFlow_" + std::to_string(ii);
                    cv::imwrite(DISFlow_name+".png",flow_rgb[flow_idx]);
                    cv::imshow(DISFlow_name, flow_rgb[flow_idx]);
                    std::string warp_name = "WrapYUV_" + std::to_string(ii);
                    cv::imwrite(warp_name+".png",rgbImageW_seq[flow_idx]);
                    cv::imshow(warp_name, rgbImageW_seq[flow_idx]);
                    std::string rgb_name = "OriginalYUV_" + std::to_string(ii);
                    cv::imwrite(rgb_name+".png",rgbImage_seq[flow_idx]);
                    cv::imshow(rgb_name, rgbImage_seq[ii + middle]);
                    cv::Mat mask_temp;
                    std::string outmask_name = "outmask_" + std::to_string(ii);
                    outmask_seq[flow_idx].convertTo(mask_temp, CV_8U, 255.0/(1.0-0.0));
                    cv::imwrite(outmask_name+".png",mask_temp);
                }
            }
            std::string rgb_name = "OriginalYUV_" + std::to_string(0);
            cv::imshow(rgb_name, rgbImage_seq[middle]);
            cv::waitKey(0);
        }

    }
    //ofstream out(outPath,ios::binary);
    return 0;
}