//
// Created by hong on 22-6-18.
//

#include "CNR_Spatial.h"
#include <iostream>
using namespace std;
void fpClear2D_small(float **fpI,float fValue, int h,int w) {
    for (int i=0; i<h;i++)
        for (int j=0; j<w;j++)
        {
            //std::cout<<"i =="<<i<<std::endl;
            //std::cout<<"j =="<<j<<std::endl;
            fpI[i][j] = fValue;
        }
}
void image_padding(float **in, float **padded_image,int width,int height,int v_size, int h_size)
{
    // -- V Direction padding -- //  coordinates is relative to padded_image
    for (int i=0; i<v_size; i++)
        for (int j=h_size; j<width+h_size;j++)
            padded_image[i][j] = in[v_size-i][j-h_size];
    for (int i=height+v_size; i<height+v_size*2; i++)
        for (int j=h_size; j<width+h_size;j++)
            padded_image[i][j] = in[height+height+v_size-i-1][j-h_size];
    // -- Inner area copy --//  coordinates is relative to original image
    for (int i=0; i<height;i++)
        for(int j=0;j<width;j++)
            padded_image[i+v_size][j+h_size]=in[i][j];
    // -- H Direction padding // coordinates id relative to padded_image
    for (int j=0;j<h_size;j++)
        for(int i=0;i<height+2*v_size;i++)
            padded_image[i][j] = padded_image[i][2*h_size-j];
    for (int j=width+h_size;j<width+h_size*2;j++)
        for(int i=0;i<height+2*v_size;i++)
            padded_image[i][j] = padded_image[i][width+2*h_size+width-j-1];

}
// ---------- guided filter on yuv domain -------------- //
float boxfilter(float **fpInput,int patch_win_v,int patch_win_h)
{
    float box_sum = 0;
    for(int j=0; j<patch_win_v;j++)
        for(int i=0; i<patch_win_h;i++)
            box_sum += fpInput[j][i];
}

void GF_YUV_Cista(int r_patch_v,   // Half size of patch Height
                  int r_patch_h,   // Half size of patch Width
                  float eps,       // Filtering parameter
                  float **fpI,     // guided image --- padded
                  float **fpP,     // noisy  image --- Padded
                  float **fpO,
                  int ratio_x,
                  int ratio_y,
                  int iWidth,int iHeight) {
//    int ratio_x = 4;
//    int ratio_y = 2;
    int patch_win_h = 2*r_patch_h/ratio_x+1;
    int patch_win_v = 2*r_patch_v/ratio_y+1;
    int N = patch_win_v*patch_win_h;
//    std::cout<<"N="<<N<<std::endl;

    //int ratio_x = 2;
    //int ratio_y = 2;
    //std::cout<<"------ log1 -------"<<std::endl;
    float **fpa = new float*[iHeight];
    float **fpb = new float*[iHeight];
    float **fpmean_a = new float*[iHeight];
    float **fpmean_b = new float*[iHeight];

    float **fpCurrent_patch_A = new float*[patch_win_v];
    float **fpCurrent_patch_B = new float*[patch_win_v];
    float **fpCurrent_patch_I = new float*[patch_win_v];
    float **fpCurrent_patch_I2 = new float*[patch_win_v];
    float **fpCurrent_patch_P = new float*[patch_win_v];
    float **fpCurrent_patch_IP = new float*[patch_win_v];
    //float **fpCurrent_patch_P2 = new float*[patch_win_v];
    for(int i=0; i<patch_win_v;i++)
    {
        fpCurrent_patch_I[i] = new float[patch_win_h];
        fpCurrent_patch_I2[i] = new float[patch_win_h];
        fpCurrent_patch_P[i] = new float[patch_win_h];
        fpCurrent_patch_IP[i] = new float[patch_win_h];
        fpCurrent_patch_A[i] = new float[patch_win_h];
        fpCurrent_patch_B[i] = new float[patch_win_h];
        //fpCurrent_patch_P2[i] = new float[patch_win_h];
    }
    for(int i=0; i<iHeight;i++)
    {
        fpa[i] = new float[iWidth];
        fpb[i] = new float[iWidth];
        fpmean_a[i] = new float[iWidth];
        fpmean_b[i] = new float[iWidth];
    }
    fpClear2D_small(fpCurrent_patch_I,0.0f,patch_win_v,patch_win_h);
    fpClear2D_small(fpCurrent_patch_I2,0.0f,patch_win_v,patch_win_h);
    fpClear2D_small(fpCurrent_patch_P,0.0f,patch_win_v,patch_win_h);
    fpClear2D_small(fpCurrent_patch_IP,0.0f,patch_win_v,patch_win_h);

    fpClear2D_small(fpa,0.0f,iHeight,iWidth);
    fpClear2D_small(fpb,0.0f,iHeight,iWidth);
    //std::cout<<"------ log2 -------"<<std::endl;
    //fpClear2D_small(fpCurrent_patch_P2,0.0f,patch_win_v,patch_win_h);
    for (int y=0; y< iHeight; y++){
        for(int x=0; x< iWidth; x++){
            int x_pad = x+r_patch_h;
            int y_pad = y+r_patch_v;
            for(int j=-r_patch_v;j<=r_patch_v; j=j+ratio_y)
                for(int i=-r_patch_h;i<=r_patch_h; i=i+ratio_x){
                    fpCurrent_patch_I[j/ratio_y+r_patch_v/ratio_y][i/ratio_x+r_patch_h/ratio_x] = fpI[y_pad+j][x_pad+i];
                    fpCurrent_patch_P[j/ratio_y+r_patch_v/ratio_y][i/ratio_x+r_patch_h/ratio_x] = fpP[y_pad+j][x_pad+i];
                    //std::cout<<"fpI, fpP = "<<fpI[]
                }
            // std::cout<<"fpI, fpP = "<<fpI[y][x]<<" ; " <<fpP[y][x]<<std::endl;
            for(int j=-r_patch_v;j<=r_patch_v; j=j+ratio_y)
                for(int i=-r_patch_h;i<=r_patch_h; i=i+ratio_x){
                    fpCurrent_patch_I2[j/ratio_y+r_patch_v/ratio_y][i/ratio_x+r_patch_h/ratio_x] = fpI[y_pad+j][x_pad+i]*fpI[y_pad+j][x_pad+i];
                    fpCurrent_patch_IP[j/ratio_y+r_patch_v/ratio_y][i/ratio_x+r_patch_h/ratio_x] = fpI[y_pad+j][x_pad+i]*fpP[y_pad+j][x_pad+i];
                }
            float meanI = boxfilter(fpCurrent_patch_I,patch_win_v,patch_win_h)/N;
            int meanI_int = int(meanI*N*33+2048)>>1;
            float meanP = boxfilter(fpCurrent_patch_P,patch_win_v,patch_win_h)/N;
            float meanIP = boxfilter(fpCurrent_patch_IP,patch_win_v,patch_win_h)/N;
            float meanII = boxfilter(fpCurrent_patch_I2,patch_win_v,patch_win_h)/N;
            float covIP = meanIP - meanI * meanP;
            float varI  = meanII - meanI * meanI;
//            if (varI<-70)
//                std::cout<<"the varI is not right"<<std::endl;
            float a = covIP / float(varI+eps);
            float b = meanP - a* meanI;
 /*           if(y==4 && x==40 ){
                std::cout<<"meanI="<<meanI<<std::endl;
                std::cout<<"meanI_int="<<meanI_int<<std::endl;
                std::cout<<"meanP="<<meanP<<std::endl;
                std::cout<<"meanIP="<<meanIP<<std::endl;
                std::cout<<"meanII="<<meanII<<std::endl;
                std::cout<<"a="<<a<<std::endl;
                std::cout<<"a="<<a<<std::endl;
                std::cout<<"b="<<b<<std::endl;
                std::cout<<"covIP="<<covIP<<std::endl;
                std::cout<<"varI="<<varI<<std::endl;
            }*/
            fpa[y][x] = a;
            fpb[y][x] = b;
        }
    }
    // ------------ calculate meanA, meanB ---------------- //
    for (int y=0; y< iHeight; y++){
        for(int x=0; x< iWidth; x++){
            fpmean_a[y][x] = fpa[y][x];
            fpmean_b[y][x] = fpb[y][x];
        }
    }

    for (int y=r_patch_v; y< iHeight-r_patch_v; y++){
        for(int x=r_patch_h; x< iWidth-r_patch_h; x++){
            for(int j=-r_patch_v;j<=r_patch_v; j=j+ratio_y)
                for(int i=-r_patch_h;i<=r_patch_h; i=i+ratio_x){
                    fpCurrent_patch_A[j/ratio_y+r_patch_v/ratio_y][i/ratio_x+r_patch_h/ratio_x] = fpa[y+j][x+i];
                    fpCurrent_patch_B[j/ratio_y+r_patch_v/ratio_y][i/ratio_x+r_patch_h/ratio_x] = fpb[y+j][x+i];
                }
            float mean_a = boxfilter(fpCurrent_patch_A,patch_win_v,patch_win_h)/N;
            float mean_b = boxfilter(fpCurrent_patch_B,patch_win_v,patch_win_h)/N;
            fpmean_a[y][x] = mean_a;
            fpmean_b[y][x] = mean_b;
        }
    }

    for (int y=0; y< iHeight; y++){
        for(int x=0; x< iWidth; x++){
            int x_pad = x+r_patch_h;
            int y_pad = y+r_patch_v;
            fpO[y][x] = fpmean_a[y][x]*fpI[y_pad][x_pad]+ fpmean_b[y][x];
/*            if(y==4 && x==40){
                std::cout<<"fpmean_a = "<<fpmean_a[y][x]<<std::endl;
                std::cout<<"part_a = "<<fpmean_a[y][x]*fpI[y_pad][x_pad]<<std::endl;
                std::cout<<"fpmean_b[y][x] = "<<fpmean_b[y][x]<<std::endl;
                std::cout<<"final result is ="<<fpO[y][x];
            }*/
        }
    }
}


void CNR_Spatial_Top(CNR_Spatial_para cnr_para,       // Filtering parameter
                     float **fpI_Ori,     // guided image --- padded
                     float **fpPU_Ori,     // noisy  image --- Padded
                     float **fpPV_Ori,     // noisy  image --- Padded
                     float **fpOU,
                     float **fpOV){
    int r_patch_v = cnr_para.r_patch_v;   // Half size of patch Height
    int r_patch_h = cnr_para.r_patch_h;   // Half size of patch Width
    float esp = cnr_para.esp;       // Filtering parameter
    int ratio_x = cnr_para.ratio_x;
    int ratio_y = cnr_para.ratio_y;
    int iWidth = cnr_para.iWidth;
    int iHeight = cnr_para.iHeight;
    int d_h = iHeight;
    int d_w = iWidth;
    int pad_size_h = r_patch_h;
    int pad_size_v = r_patch_v;
    float **fpI_padded = new float *[d_h+2*pad_size_v];
    float **fpPU_padded = new float *[d_h+2*pad_size_v];
    float **fpPV_padded = new float *[d_h+2*pad_size_v];
    for(int i=0;i<d_h+2*pad_size_v;i++)
    {
        fpI_padded[i] = new float[d_w+2*pad_size_h];
        fpPU_padded[i] = new float[d_w+2*pad_size_h];
        fpPV_padded[i] = new float[d_w+2*pad_size_h];
        //denoised_padded[i] = new float[d_w+2*pad_size_h];
    }

    for (int i=0; i<d_h+2*pad_size_v; i++)
    {
        for(int j=0; j<d_w+2*pad_size_h; j++)
        {
            fpI_padded[i][j] = 0;
            fpPU_padded[i][j] = 0;
            fpPV_padded[i][j] = 0;
        }
    }
    image_padding(fpI_Ori,fpI_padded,iWidth,iHeight,pad_size_v,pad_size_h);
    image_padding(fpPU_Ori,fpPU_padded,iWidth,iHeight,pad_size_v,pad_size_h);
    image_padding(fpPV_Ori,fpPV_padded,iWidth,iHeight,pad_size_v,pad_size_h);
    GF_YUV_Cista(r_patch_v,r_patch_h,esp,fpI_padded,fpPU_padded,fpOU,ratio_x,ratio_y,d_w, d_h);
    GF_YUV_Cista(r_patch_v,r_patch_h,esp,fpI_padded,fpPV_padded,fpOV,ratio_x,ratio_y,d_w, d_h);
    // －－－－－ Delete the Temporal Pointers －－－－－－ //
    for (int i=0;i<d_h+2*pad_size_v;i++){
        delete [] fpI_padded[i];
        delete [] fpPU_padded[i];
        delete [] fpPV_padded[i];
    }
    delete [] fpI_padded;
    delete [] fpPU_padded;
    delete [] fpPV_padded;
}
