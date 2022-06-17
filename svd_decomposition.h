//
// Created by hong on 22-5-31.
//

#ifndef INC_3DNR_OF_SVD_DECOMPOSITION_H
#define INC_3DNR_OF_SVD_DECOMPOSITION_H
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <cmath>
void compute_pca_svd(
        float  ** Ain,
        float ** U,
        float ** V,
        float * W,
        int row,
        int cols
        );
#endif //INC_3DNR_OF_SVD_DECOMPOSITION_H