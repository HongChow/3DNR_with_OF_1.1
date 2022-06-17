//
// Created by hong on 22-6-1.
//

#ifndef INC_3DNR_OF_PCADENOISE_H
#define INC_3DNR_OF_PCADENOISE_H
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <cmath>
#include "svd_decomposition.h"
__inline int CLIP(float indata,float min, float max)
{
    if (indata<min)
        return min;
    else if (indata>max)
        return max;
    else
        return indata;
}
void PCADenoise(
        float ** input_3D_Blocks,
        float * mean_noisy_3D,
        float fSigma,
        float fRMult,
        int rows,
        int cols,
        float ** output_3D_Blocks
);
#endif //INC_3DNR_OF_PCADENOISE_H
