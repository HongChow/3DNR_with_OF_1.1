//
// Created by hong on 22-6-18.
//

#ifndef INC_3DNR_OF_CNR_SPATIAL_H
#define INC_3DNR_OF_CNR_SPATIAL_H

struct CNR_Spatial_para{
    int iWidth;
    int iHeight;
    int r_patch_v;
    int r_patch_h;
    float esp;
    int ratio_x;// ---frame nums used = 2*radius_Tem
    int ratio_y;
};
void image_padding(float **in,
                   float **padded_image,
                   int width,
                   int height,
                   int v_size,
                   int h_size);
void GF_YUV_Cista(int r_patch_v,   // Half size of patch Height
                  int r_patch_h,   // Half size of patch Width
                  float esp,       // Filtering parameter
                  float **fpI,     // guided image --- padded
                  float **fpP,     // noisy  image --- Padded
                  float **fpO,
                  int ratio_x,
                  int ratio_y,
                  int iWidth,int iHeight);
void CNR_Spatial_Top(CNR_Spatial_para cnr_para,       // Filtering parameter
                     float **fpI_Ori,     // guided image --- padded
                     float **fpPU_Ori,     // noisy  image --- Padded
                     float **fpPV_Ori,     // noisy  image --- Padded
                     float **fpOU,
                     float **fpOV);
#endif //INC_3DNR_OF_CNR_SPATIAL_H
