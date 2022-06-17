//
// Created by hong on 22-5-24.
//
#include "Spatial_Temperal_Denoise.h"
#include "PCADenoise.h"
#include "omp.h"
//#define DEBUG=1
//#define TIME_CAL
int order_increasing_diffs(const void *pVoid1, const void *pVoid2)
{
    struct diffpatches *node1, *node2;

    node1=(struct diffpatches *) pVoid1;
    node2=(struct diffpatches *) pVoid2;

    if (node1->diff < node2->diff) return -1;
    if (node1->diff > node2->diff) return 1;
    return 0;
}

cv::Mat crop_img(cv::Mat img_ori,int j, int i, int radius){
    cv::Mat roi_img;
    cv::Rect roi_locations;
    roi_locations.x = i-radius;
    roi_locations.y = j-radius;
    roi_locations.width = radius*2+1;
    roi_locations.height = radius*2+1;
    cv::Mat img_part = img_ori(roi_locations);
    img_part.copyTo(roi_img);
    return roi_img;
}


std::vector<struct patch_selected>  get_block_differences(cv::Mat rgbImageW_seq[],cv::Mat current_rgbImage,float ** R_Warped[],float **G_Warped[],float ** B_Warped[],int y,int x,int jmin,int jmax,int imin,int imax,int radius_block,int nums_sequence,int & npts){
    struct diffpatches *pdiff=new struct diffpatches[(imax-imin+1)*(jmax-jmin+1)];
    std::vector<struct patch_selected> selected_blocks;
    std::vector<cv::Mat> rgbVects(3), rgbImage_W_Vects(3),block_temp_Vects(3);
    //std::vector<cv::Mat> rgbVects(3),block_temp_Vects(3);
    cv::split(current_rgbImage, rgbVects);
    //std::vector<std::vector<cv::Mat >> rgbImage_W_Vects(nums_sequence);
    int ikt = 0;
    float fThresh_diff = 0; // -----------------TODO : Set the threshold value ----------------//
    //for (int t = 0; t < nums_sequence; t++) {
    //    cv::split(rgbImageW_seq[t], rgbImage_W_Vects[t]);
    //}
//#pragma omp parallel
//    {
//#pragma  omp for
//    omp_set_num_threads(8);
//#pragma omp parallel for schedule(static,250)
/*
        for (int j = jmin; j <= jmax; j++) {
            for (int i = imin; i <= imax; i++, ikt++) {
                float dist = 0.0;

//#pragma omp parallel for reduction(+:dist)

                for (int t = 0; t < nums_sequence; t++) {
                    for (int ii = 0; ii < 3; ii++) {
                        for (int m = -radius_block; m <= radius_block; m++) {
                            for (int n = -radius_block; n <= radius_block; n++) {
//#pragma  omp atomic
                                dist += (float(rgbImage_W_Vects[t][ii].at<uchar>(y + m, x + n)) -
                                         float(rgbImage_W_Vects[t][ii].at<uchar>(j + m, i + n))) *
                                        (float(rgbImage_W_Vects[t][ii].at<uchar>(y + m, x + n))
                                         - float(rgbImage_W_Vects[t][ii].at<uchar>(j + m, i + n)));
                            }
                        }
                    }
                }

                for (int ii = 0; ii < 3; ii++) {
                    for (int m = -radius_block; m <= radius_block; m++) {
                        for (int n = -radius_block; n <= radius_block; n++) {
//#pragma  omp atomic
                                      dist += (float(rgbVects[ii].at<uchar>(y + m, x + n)) -
                                     float(rgbVects[ii].at<uchar>(j + m, i + n))) *
                                    (float(rgbVects[ii].at<uchar>(y + m, x + n)) -
                                     float(rgbVects[ii].at<uchar>(j + m, i + n)));
                        }
                    }
                }

                pdiff[ikt].diff = dist;
                pdiff[ikt].i = i;
                pdiff[ikt].j = j;
            }
        }

*/
    //}
//#pragma omp parallel for


   //auto beforeFetch3DTime2 = std::chrono::steady_clock::now();

    for (int j = jmin; j <= jmax; j++) {
        for (int i = imin; i <= imax; i++, ikt++) {
            float dist2 = 0.0;
            for (int t = 0; t < nums_sequence-1; t++) {
                    for (int m = -radius_block; m <= radius_block; m++) {
                        for (int n = -radius_block; n <= radius_block; n++) {
                            dist2 += (R_Warped[t][y+m][x+n]-R_Warped[t][j+m][i+n])*(R_Warped[t][y+m][x+n]-R_Warped[t][j+m][i+n]);
                            dist2 += (G_Warped[t][y+m][x+n]-G_Warped[t][j+m][i+n])*(G_Warped[t][y+m][x+n]-G_Warped[t][j+m][i+n]);
                            dist2 += (B_Warped[t][y+m][x+n]-B_Warped[t][j+m][i+n])*(B_Warped[t][y+m][x+n]-B_Warped[t][j+m][i+n]);
                        }
                    }

            }
            for (int ii = 0; ii < 3; ii++) {
                for (int m = -radius_block; m <= radius_block; m++) {
                    for (int n = -radius_block; n <= radius_block; n++) {
                        dist2 += (float(rgbVects[ii].at<uchar>(y + m, x + n)) -
                                 float(rgbVects[ii].at<uchar>(j + m, i + n))) *
                                (float(rgbVects[ii].at<uchar>(y + m, x + n)) -
                                 float(rgbVects[ii].at<uchar>(j + m, i + n)));
                    }
                }
            }
            pdiff[ikt].diff = dist2;
            pdiff[ikt].i = i;
            pdiff[ikt].j = j;
        }
    }

    //auto beforeFetch3DTime = std::chrono::steady_clock::now();
    //double duration_millsecond_fetch3D2 = std::chrono::duration<double, std::milli>(beforeFetch3DTime - beforeFetch3DTime2).count();
    //std::cout<<"inner fetch 3D2 time = "<<duration_millsecond_fetch3D2<<std::endl;
    /*

    for (int j = jmin; j <= jmax; j++) {
        for (int i = imin; i <= imax; i++, ikt++) {
            float dist = 0.0;
            for (int t = 0; t < nums_sequence-1; t++) {
                if (j == jmin and i == imin)
                    cv::split(rgbImageW_seq[t], rgbImage_W_Vects);
                for (int ii = 0; ii < 3; ii++) {
                    for (int m = -radius_block; m <= radius_block; m++) {
                        for (int n = -radius_block; n <= radius_block; n++) {
                            dist += (float(rgbImage_W_Vects[ii].at<uchar>(y + m, x + n)) -
                                     float(rgbImage_W_Vects[ii].at<uchar>(j + m, i + n))) *
                                    (float(rgbImage_W_Vects[ii].at<uchar>(y + m, x + n))
                                     - float(rgbImage_W_Vects[ii].at<uchar>(j + m, i + n)));
                        }
                    }
                }
            }
            for (int ii = 0; ii < 3; ii++) {
                for (int m = -radius_block; m <= radius_block; m++) {
                    for (int n = -radius_block; n <= radius_block; n++) {
                        dist += (float(rgbVects[ii].at<uchar>(y + m, x + n)) -
                                 float(rgbVects[ii].at<uchar>(j + m, i + n))) *
                                (float(rgbVects[ii].at<uchar>(y + m, x + n)) -
                                 float(rgbVects[ii].at<uchar>(j + m, i + n)));
                    }
                }
            }

            pdiff[ikt].diff = dist;
            pdiff[ikt].i = i;
            pdiff[ikt].j = j;
        }
    }*/
    //auto afterFetch3DTime = std::chrono::steady_clock::now();
    //double duration_millsecond_fetch3D = std::chrono::duration<double, std::milli>(afterFetch3DTime - beforeFetch3DTime).count();
    //std::cout<<"inner fetch 3D time = "<<duration_millsecond_fetch3D<<std::endl;
    qsort(pdiff, (imax-imin+1)*(jmax-jmin+1), sizeof(struct diffpatches), order_increasing_diffs);
    bool static_flag = true;
    struct patch_selected patch_temp;
    for (int ikt_idx = 0;(ikt_idx<((imax-imin+1)*(jmax-jmin+1)))&&static_flag;ikt_idx++){
        float diff_this = pdiff[ikt_idx].diff;
        //if(ikt_idx==168){
        //    std::cout<<"this is the end ?"<<std::endl;
        //}

        for(int idx = 0;idx<nums_sequence;idx++){
            // ---------------- TODO : using confidence mask of optical flow, exclude the possible unreliable patches ----------------- //
            //std::cout<<"ikt_idx="<<ikt_idx<<"__"<<"idx="<<idx<<std::endl;
            patch_temp.i=pdiff[ikt_idx].i;
            patch_temp.j=pdiff[ikt_idx].j;
            patch_temp.diff = diff_this;
            patch_temp.seq_idx=idx;
            if (idx<3)
                patch_temp.block_img = crop_img(rgbImageW_seq[idx],pdiff[ikt_idx].j, pdiff[ikt_idx].i,radius_block);
            else
                if (idx==3)
                    patch_temp.block_img = crop_img(current_rgbImage,pdiff[ikt_idx].j, pdiff[ikt_idx].i,radius_block);
                else // idx>3
                    patch_temp.block_img = crop_img(rgbImageW_seq[idx-1],pdiff[ikt_idx].j, pdiff[ikt_idx].i,radius_block);
            std::vector<cv::Mat> rgbVects_roi_mat(3);
            cv::split(patch_temp.block_img, rgbVects_roi_mat);
            patch_temp.block_img_vects = rgbVects_roi_mat;
            selected_blocks.push_back(patch_temp);
            npts++;
            if ((npts>=83)&& diff_this>fThresh_diff)
                static_flag = false;
        }
    }
    //auto afterGet3DTime = std::chrono::steady_clock::now();
    //double duration_millsecond_Get3D = std::chrono::duration<double, std::milli>( afterGet3DTime- afterFetch3DTime).count();
    //std::cout<<"inner Get 3D time = "<<duration_millsecond_Get3D<<std::endl;
    delete[] pdiff;
    return  selected_blocks;
}

void PreCalcuPatches(std::vector<struct patch_selected>Blocks_Selected,int npts,int channels, int radius_block,float *vmean, float &stdAvg){
    int iSize = (radius_block*2+1)*(radius_block*2+1);
    double fValueAux = 0.0f;
    stdAvg=0.0f;
    for(int ii=0;ii<channels;ii++){
        double mean=0.0f, std=0.0f;
        for (int idx = 0; idx<npts; idx++){
            for(int m=0; m<radius_block*2+1; m++)
                for(int n=0; n<radius_block*2+1; n++) {
                    fValueAux = (float) Blocks_Selected[idx].block_img_vects[ii].at<uchar>(m, n);
                    mean += fValueAux;
                    std += (fValueAux * fValueAux);
                }
            }

        mean /= (double)(npts*iSize);
        std /= (double) (npts * iSize);
        std -= (mean*mean);
        std = sqrt(std);
        vmean[ii]=mean;
        stdAvg += (float) std;
    }
    stdAvg /= channels;
}


void patch_avg_denoise(std::vector<struct patch_selected> Blocks_Selected,std::vector<struct patch_selected> & blocks_denoised,float *vmean,int block_size){

    for (int i=0; i<Blocks_Selected.size(); i++){
        for(int c=0; c<3; c++)
        for(int m = 0; m<block_size; m++)
            for(int n=0; n< block_size; n++){
                blocks_denoised[i].block_img_vects[c].at<uchar>(m,n) = (int)(vmean[c]);
            }
    }
}

void Calculate_Mean_Blocks(std::vector<struct patch_selected> Blocks_Selected,int channels,std::vector<cv::Mat> & block_mean_vects,int rows,int cols){
    int total_block_nums = Blocks_Selected.size();
    for (int c = 0; c< channels; c++){
        for(int m = 0; m< rows; m++)
            for(int n = 0; n< cols; n++) {
                double sum_c =0;
                for (int block_id = 0; block_id < total_block_nums; block_id++) {
                    sum_c += (double)Blocks_Selected[block_id].block_img_vects[c].at<uchar>(m,n);
                }
                block_mean_vects[c].at<float>(m,n) = (float)sum_c/total_block_nums;
            }
    }

}

// ----------- Aggregation Operation ------------ //
void Aggregation(std::vector<struct patch_selected> Blocks_Denoised,float ** Weights_Mask,std::vector<cv::Mat>& OutPutImages,int y, int x, int radius){
    int start_y = y - radius;
    int start_x = x - radius;
    for (int i = 0; i< Blocks_Denoised.size();i++){
        if(Blocks_Denoised[i].seq_idx==3){
            for(int m = 0; m<radius; m++)
                for(int n = 0; n<radius; n++){
                    // ---------- for debug -------------//
                    /*if (start_y+m ==18 && start_x+n==8){
                        std::cout<<"debug_weights = "<< Weights_Mask[18][10]<<std::endl;
                        //std::
                        std::cout<<"OutPutImages = "<<OutPutImages[0].at<float>(18,8)<<std::endl;
                    }*/
                    Weights_Mask[start_y+m][start_x+n] += 1;
                    for(int c = 0; c<OutPutImages.size(); c++){
                        //float temp = (float)Blocks_Denoised[i].block_img_vects[c].at<uchar>(start_y+m,start_x+n);
                        OutPutImages[c].at<float>(start_y+m,start_x+n)+=(float)Blocks_Denoised[i].block_img_vects[c].at<uchar>(m,n);
                    }
                }
        }
    }
}
void NormalizeOutput(cv::Mat current_rgbImage, float **Weights_Mask,std::vector<cv::Mat> OutPutImages,cv::Mat &OutPutFrame,int height,int width)
{
    // Normalize values
    std::vector<cv::Mat> current_rgbFrames(3);
    std::vector<cv::Mat> output_rgbFrames(3);
    for (int c=0;c<3;c++)
        output_rgbFrames[c].create(height,width,CV_8U);
    cv::split(current_rgbImage,current_rgbFrames);
    for(int j=0; j < height; j++)
        for(int i=0; i < width; i++) {
            for (int c=0; c < 3; c++) {
                float denoised_temp;
                if (Weights_Mask[j][i] > 0.0) {
                    denoised_temp = OutPutImages[c].at<float>(j, i) / Weights_Mask[j][i];
                    //OutPutImages[c].at<float>(j, i) /= Weights_Mask[j][i];
                } else {
                    denoised_temp = (float) current_rgbFrames[c].at<uchar>(j, i);
                    //OutPutImages[c].at<float>(j, i) = (float) current_rgbFrames[c].at<uchar>(j, i);
                }
                output_rgbFrames[c].at<uchar>(j, i) =  (uchar)denoised_temp;
                //output_rgbFrames[c].at<uchar>(j, i) = (uchar)OutPutImages[c].at<float>(j, i);
            }
        }
    cv::merge(output_rgbFrames,OutPutFrame);
}
//cv::Mat Spatial_Temperal_Denoise(cv::Mat rgbImageW_seq[],cv::Mat current_rgbImage,denoise_para param) {
void Spatial_Temperal_Denoise(cv::Mat rgbImageW_seq[],cv::Mat current_rgbImage,float ** R_Warped[],float **G_Warped[],float ** B_Warped[],denoise_para param,float ** Weights_Mask,std::vector<cv::Mat>& OutPutImages) {
        // -------------- Get Similiar 3D Blocks ------------------//
        int width = param.width;
        int height = param.height;
        int radius_block = param.radius_block;
        int radius_search = param.radius_search;
        int nums_sequence = param.radius_Tem * 2 + 1;
        int block_size = param.radius_block*2 + 1;
        int channels = param.channels;
        float FlatPar = param.oflat;
        float fSigma = param.osigma;
        float fRMult = param.ofpca;
        std::vector<cv::Mat> block_mean_vects(channels);
        // -----------allocate memory for block_mean_vects-------------- //
        for (int i=0;i<channels;i++){
            block_mean_vects[i].create(block_size,block_size,CV_32FC1);
        }
        for (int y = radius_block; y < height - radius_block - 1; y++) {
            std::cout<<"the coordinates is finished @ y= "<<y <<std::endl;
            int jmin = MAX(y - radius_search, radius_block);
            int jmax = MIN(y + radius_search, height - 1 - radius_block);
            float total_Fetch3D_Time = 0.0f,total_PCA_Time = 0.0f;
            for (int x = radius_block; x < width - radius_block - 1; x++) {
                int imin = MAX(x - radius_search, radius_block);
                int imax = MIN(x + radius_search, width - 1 - radius_block);
                // ------------- Get sorted 3D Blocks select the valid points and get rid of unsuitable patches-------------- //
                int npts = 0;
#ifdef DEBUG
                if(y==235)
                    std::cout<<"--------- x =  "<<x<<" ----------"<<std::endl;
#endif
                auto beforeFetch3DTime = std::chrono::steady_clock::now();
                std::vector<struct patch_selected> Blocks_Selected = get_block_differences(rgbImageW_seq,
                                                                                           current_rgbImage, R_Warped,G_Warped,B_Warped,y, x, jmin,
                                                                                           jmax, imin, imax,
                                                                                           radius_block, nums_sequence,
                                                                                           npts);
                auto afterFetch3DTime = std::chrono::steady_clock::now();
                double duration_millsecond = std::chrono::duration<double, std::milli>(afterFetch3DTime - beforeFetch3DTime).count();
                //std::cout <<"get_block_differences"<< duration_millsecond << "毫秒" << std::endl;
                total_Fetch3D_Time+=duration_millsecond;
                std::vector<struct patch_selected> Blocks_Denoised;
                for (int i = 0; i < Blocks_Selected.size(); i++) {
                    Blocks_Denoised.push_back(Blocks_Selected[i]);
                }
                //cv::imshow("current_rgbImage", current_rgbImage);
                //cv::waitKey(0);Calculate_Mean_Blocks
                //------- for debug and save blocks to show -------
                //std::cout<<"Blocks_Selected.size() : "<<Blocks_Selected.size()<<std::endl;
#ifdef block_denoise_debug
                if (y==5&&x==140)
                for (int i=0;i<Blocks_Selected.size();i++){
                    std::cout<<"In Saving 3d Blocks" <<".... i = "<<i<<std::endl;
                    std::string Blocks_Selected_File_name = "/media/hong/62CC6F80CC6F4D7B/3DNR/Implementation/3DNR_with_OF/build/Blocks_File/Blocks_Selected_G_" + std::to_string(i);
                    Blocks_Selected_File_name = Blocks_Selected_File_name + "_.txt";
                    std::ofstream Blocks_Selected_File;
                    Blocks_Selected_File.open(Blocks_Selected_File_name);
                    for (int row = 0; row<radius_block*2+1; row++)
                        for (int col= 0; col<radius_block*2+1; col++){
                            Blocks_Selected_File<<(int)Blocks_Selected[i].block_img_vects[1].at<uchar>(row,col);
                            Blocks_Selected_File<< " ";
                            if (col==radius_block*2)
                                Blocks_Selected_File<<std::endl;
                        }
                    Blocks_Selected_File.close();
                    std::string block_name = "/home/hong/3DNR/3DNR_with_OF/build/3D_Blocks/searched_3D_block"+std::to_string(i)+".png";
                    //std::string block_name = "/media/hong/62CC6F80CC6F4D7B/3DNR/Implementation//3DNR_with_OF/build/3D_Blocks/searched_3D_block"+std::to_string(i)+".png";
                    //std::cout<<"Block_Name = "<<block_name<<std::endl;
                    cv::Mat Resize3D;
                    cv::resize(Blocks_Selected[i].block_img,Resize3D,cv::Size(100,100));
                    //std::cout<<"diff = "<<Blocks_Selected[i].diff<<std::endl;
                    //cv::putText(Resize3D,std::to_string(Blocks_Selected[i].diff),cv::Point(50,50), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 255));
                    cv::imwrite(block_name,Resize3D);
                    //cv::imshow(block_name, Resize3D);
                    //cv::waitKey(0);
                    //cv::imwrite(block_name,Blocks_Selected[i].block_img);
                }
#endif
                // ------------- Pre-Calculate 3D Blocks Selected ------------------ //
                float *vmean = new float[channels];
                float stdAvg;
#ifdef DEBUG
                if(y==2 && x==346)
                    std::cout<<"--------- log 2 ----------"<<std::endl;
#endif
                PreCalcuPatches(Blocks_Selected, npts, channels, radius_block, vmean, stdAvg);

#ifdef DEBUG
                if(y==2 && x==346)
                    std::cout<<"--------- log 3 ----------"<<std::endl;
#endif
                // consider if the blocks are in flat areas //
                if (stdAvg < FlatPar * fSigma) { // is flat set the average mode
                   // std::cout<<"with avg"<<std::endl;
                 // if (stdAvg < FlatPar * fSigma) { // is flat set the average mode //
                    patch_avg_denoise(Blocks_Selected, Blocks_Denoised, vmean, block_size);
#ifdef DEBUG
                    if(y==2 && x==346)
                        std::cout<<"--------- log 4 ----------"<<std::endl;
#endif
                } else { // ------------ Do PCA Based Denoise ------------------- //
                    // ----------- Mean Blocks ---------- //

                    int points_num = block_size * block_size * channels;
                    //Calculate_Mean_Blocks(Blocks_Selected,channels,block_mean_vects, npts,points_num);
                    Calculate_Mean_Blocks(Blocks_Selected, channels, block_mean_vects, block_size, block_size);
#ifdef DEBUG
                    if(y==2 && x==346)
                        std::cout<<"--------- log 5 ----------"<<std::endl;
#endif
                    // --------- Convert 3D noisy Blocks into 2-Dimension Arrays ---------- //
                    float **input_3D_Blocks = new float *[npts];
                    float **output_3D_Blocks = new float *[npts];
                    float *mean_3D_Blocks = new float[points_num];
                    for (int i = 0; i < npts; i++) {
                        input_3D_Blocks[i] = new float[points_num];
                        output_3D_Blocks[i] = new float[points_num];
                    }
                    for (int i = 0; i < npts; i++)
                        for (int j = 0; j < points_num; j++) {
                            output_3D_Blocks[i][j] = 0.0f;
                        }
                    for (int i = 0; i < npts; i++)
                        for (int m = 0; m < block_size; m++)
                            for (int n = 0; n < block_size; n++)
                                for (int c = 0; c < channels; c++) {
                                    input_3D_Blocks[i][((m * block_size) + n) * channels + c] =
                                            (float) Blocks_Selected[i].block_img_vects[c].at<uchar>(m, n) -
                                            block_mean_vects[c].at<float>(m, n);
                                }
                    // --------- Convert 2D Mean Blocks into 1-Dimension Vectors ---------- //
                    for (int m = 0; m < block_size; m++)
                        for (int n = 0; n < block_size; n++)
                            for (int c = 0; c < channels; c++) {
                                mean_3D_Blocks[((m * block_size) + n) * channels + c] = block_mean_vects[c].at<float>(m,
                                                                                                                      n);
                            }
                    auto beforePCATime = std::chrono::steady_clock::now();
                    PCADenoise(input_3D_Blocks,
                               mean_3D_Blocks,
                               fSigma,
                               fRMult,
                               npts,
                               points_num,
                               output_3D_Blocks);
                    auto afterPCATime = std::chrono::steady_clock::now();
                    double duration_millsecondPCA = std::chrono::duration<double, std::milli>(afterPCATime - beforePCATime).count();
                    total_PCA_Time+=duration_millsecondPCA;
                    //std::cout <<"PCA:"<< duration_millsecondPCA << "毫秒" << std::endl;
#ifdef DEBUG
                    if(y==2 && x==346)
                        std::cout<<"--------- log 6 ----------"<<std::endl;
#endif
                    // --------- Convert back from 2D-Dimension Denoised Arrays into 3D denoised Blocks ----------- //
                    for (int i = 0; i < npts; i++)
                        for (int m = 0; m < block_size; m++)
                            for (int n = 0; n < block_size; n++) {
                                for (int c = 0; c < channels; c++) {
                                    Blocks_Denoised[i].block_img_vects[c].at<uchar>(m, n) = (uchar) output_3D_Blocks[i][
                                            ((m * block_size) + n) * channels + c];
                                    //Blocks_Denoised[i].block_img_vects[c].at<float>(m,n) = output_3D_Blocks[i][((m * block_size) + n) * channels + c];
                                }
                                cv::merge(Blocks_Denoised[i].block_img_vects,Blocks_Denoised[i].block_img );
                            }
                    //std::cout<<"the coordinates is finished @ y= "<<y <<"  and @x = "<<x<<" "<<std::endl;
                    // ----------------------------- Debug the output denoised Blocks ------------------------------ //
                    //float before_assign = output_3D_Blocks[0][(3) * 3 + 0];
                    //float temp = Blocks_Denoised[0].block_img_vects[0].at<float>(0,3);
                    // ----------------------- for debug ----------------------- //
#ifdef block_denoise_debug
                    if (y==5&&x==140)
                    for (int i=0;i<Blocks_Denoised.size();i++) {
                        std::cout << "In Saving 3d Blocks" << ".... i = " << i << std::endl;
                        std::string Blocks_Denoised_File_name = "/home/hong/3DNR/3DNR_with_OF/build/Blocks_File/Blocks_Denoised_G_" + std::to_string(i);
                        //std::string Blocks_Denoised_File_name = "/media/hong/62CC6F80CC6F4D7B/3DNR/Implementation/3DNR_with_OF/build/Blocks_File/Blocks_Denoised_G_" + std::to_string(i);
                        Blocks_Denoised_File_name = Blocks_Denoised_File_name + "_.txt";
                        std::ofstream Blocks_Denoised_File;
                        Blocks_Denoised_File.open(Blocks_Denoised_File_name);
                        for (int row = 0; row < radius_block * 2 + 1; row++)
                            for (int col = 0; col < radius_block * 2 + 1; col++) {
                                Blocks_Denoised_File << (int) Blocks_Denoised[i].block_img_vects[1].at<uchar>(row, col);
                                Blocks_Denoised_File << " ";
                                if (col == radius_block * 2)
                                    Blocks_Denoised_File << std::endl;
                            }
                        Blocks_Denoised_File.close();
                        //std::string block_name = "/media/hong/62CC6F80CC6F4D7B/3DNR/Implementation//3DNR_with_OF/build/3D_Blocks/Denoised_Block_"+std::to_string(i)+".png";
                        std::string block_name = "/home/hong/3DNR/3DNR_with_OF/build/3D_Blocks/Denoised_Block_"+std::to_string(i)+".png";
                        //std::cout<<"Block_Name = "<<block_name<<std::endl;
                        cv::Mat Resize3D_Denoised;
                        cv::resize(Blocks_Denoised[i].block_img,Resize3D_Denoised,cv::Size(100,100));
                        //std::cout<<"diff = "<<Blocks_Selected[i].diff<<std::endl;
                        //cv::putText(Resize3D,std::to_string(Blocks_Selected[i].diff),cv::Point(50,50), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 255));
                        cv::imwrite(block_name,Resize3D_Denoised);
                    }
#endif
                    for (int i = 0; i < npts; i++) {
                        delete [] input_3D_Blocks[i];
                        delete [] output_3D_Blocks[i];
                    }
                    delete[] input_3D_Blocks;
                    delete[] output_3D_Blocks;
                    delete[] mean_3D_Blocks;
                }
                Aggregation( Blocks_Denoised,Weights_Mask, OutPutImages, y,  x, radius_block);
                delete vmean;
            }
            std::cout <<"avrage PCA time :"<< total_PCA_Time/(height-radius_block-1-radius_block)<< "毫秒" << std::endl;
            std::cout <<"avrage Fetch 3D time :"<< total_Fetch3D_Time/(height-radius_block-1-radius_block)<< "毫秒" << std::endl;
        }


}
