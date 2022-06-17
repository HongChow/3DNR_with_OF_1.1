//
// Created by hong on 22-5-14.
//

const float hBin = 0.5f;
#include "optical_flow_related.h"
const float hbin = 0.5;
static int neumann_bc(int x, int nx)
{
    if(x < 0) {
        x = 0;
        //*out = true;
    } else if (x >= nx) {
        x = nx - 1;
        //*out = true;
    }

    return x;
}
static double cubic_interpolation_cell (
        double v[4],  //interpolation points
        double x      //point to be interpolated
)
{
    return  v[1] + 0.5 * x * (v[2] - v[0] +
                              x * (2.0 *  v[0] - 5.0 * v[1] + 4.0 * v[2] - v[3] +
                                   x * (3.0 * (v[1] - v[2]) + v[3] - v[0])));
}


static double bicubic_interpolation_cell (
        double p[4][4], //array containing the interpolation points
        double x,       //x position to be interpolated
        double y        //y position to be interpolated
)
{
    double v[4];
    v[0] = cubic_interpolation_cell(p[0], y);
    v[1] = cubic_interpolation_cell(p[1], y);
    v[2] = cubic_interpolation_cell(p[2], y);
    v[3] = cubic_interpolation_cell(p[3], y);
    return cubic_interpolation_cell(v, x);
}


cv::Mat WarpFrame(cv::Mat rgbImage,cv::Mat flow,int width, int height){
    cv::Mat flow_uv[2];
    cv::split(flow, flow_uv);
    std::vector<cv::Mat> rgbVects(3),rgbImage_W_Vects(3);
    //cv::Mat rgbVects[3];
    //cv::Mat rgbImage_W_Vects[3];
    //std::cout<<rgbImage.channels()<<" -             -- -        -"<<std::endl;
    cv::split(rgbImage,rgbVects);
    cv::Mat rgbImage_W;
    rgbImage_W.create(height,width,CV_8UC3);
    //rgbImage.copyTo(rgbImage_W);
    cv::split(rgbImage_W,rgbImage_W_Vects);
    float Intp_rgb[3];
    int nx = width;
    int ny = height;
    int x, y, mx, my, dx, dy, ddx, ddy;
    for(int j=0; j < height; j++) {
    //   std::cout<<"j = ------------- "<<j<<std::endl;
        for (int i = 0; i < width; i++) {
            //if (j==101 and i==263)
            //    std::cout<<"i(x) = ------------- "<<i<<" -------------j(y) = ------------" <<j<<"  "<<std::endl;
            float p[2] = {i + flow_uv[0].at<float>(j, i), j + flow_uv[1].at<float>(j, i)};
            int sx = (p[0] < 0) ? -1 : 1;
            int sy = (p[1] < 0) ? -1 : 1;
            float uu = p[0];
            float vv = p[1];
            //bool out[1] = {false};
            // ---------- flow  adjust new coordinates boarder fix ------------ //
            x = neumann_bc((int) uu, nx);
            y = neumann_bc((int) vv, ny);
            mx = neumann_bc((int) uu - sx, nx);
            my = neumann_bc((int) vv - sx, ny);
            dx = neumann_bc((int) uu + sx, nx);
            dy = neumann_bc((int) vv + sy, ny);
            ddx = neumann_bc((int) uu + 2 * sx, nx);
            ddy = neumann_bc((int) vv + 2 * sy, ny);
            //if (j==287 and i==351)
            //    std::cout<<"------------- log2.0 ----------------\n";
            for (int ii = 0; ii < 3; ii++) {
             //   if (j==287 and i==351 and ii==2)
            //        std::cout<<"------------- log3.0 ----------------\n";
                const float p11 = rgbVects[ii].at<uchar>(my, mx);
                const float p12 = rgbVects[ii].at<uchar>(my, x);
                const float p13 = rgbVects[ii].at<uchar>(my, dx);
                const float p14 = rgbVects[ii].at<uchar>(my, ddx);

                const float p21 = rgbVects[ii].at<uchar>(y, mx);
                const float p22 = rgbVects[ii].at<uchar>(y, x);
                const float p23 = rgbVects[ii].at<uchar>(y, dx);
                const float p24 = rgbVects[ii].at<uchar>(y, ddx);

                const float p31 = rgbVects[ii].at<uchar>(dy, mx);
                const float p32 = rgbVects[ii].at<uchar>(dy, x);
                const float p33 = rgbVects[ii].at<uchar>(dy, dx);
                const float p34 = rgbVects[ii].at<uchar>(dy, ddx);

                const float p41 = rgbVects[ii].at<uchar>(ddy, mx);
                const float p42 = rgbVects[ii].at<uchar>(ddy, x);
                const float p43 = rgbVects[ii].at<uchar>(ddy, dx);
                const float p44 = rgbVects[ii].at<uchar>(ddy, ddx);

                double pol[4][4] = {
                        {p11, p21, p31, p41},
                        {p12, p22, p32, p42},
                        {p13, p23, p33, p43},
                        {p14, p24, p34, p44}
                };

                Intp_rgb[ii] = bicubic_interpolation_cell(pol, uu - x, vv - y);
                //if (j==101 and i==263 and ii==0) {
                    //std::cout << "------------- log4.0 ----------------\n";
                //    std::cout << "~~~~~~~~~~~~~~~~ "<< Intp_rgb[ii]<<" ~~~~~~~~~~~~~~~~~\n";
                //}
                rgbImage_W_Vects[ii].at<uchar>(j, i) = CLIP(int(Intp_rgb[ii] + 0.5),0,255);
                //rgbImage_W_Vects[ii].at<uchar>(j, i) = int(Intp_rgb[ii] + 0.5);
                //if (j==287 and i==351 and ii==2)
                //    std::cout<<"------------- log4.1 ----------------\n";
            }
        }
    }
    cv::merge(rgbImage_W_Vects,rgbImage_W);
    //if (j==287 and i==351 and ii==2)
    //    std::cout<<"------------- log4.2 ----------------\n";
    return rgbImage_W;
}

cv::Mat calculate_divergence(cv::Mat flow,int width, int height){
    cv::Mat flow_uv[2];
    cv::split(flow, flow_uv);
    cv::Mat divergence;
    divergence.create(height,width,CV_32FC1);
    for (int i = 1; i < height-1; i++) // --- y direction --- //
    {
        int p, p1, p2;
        double v1x, v2y;

        // ----- inner area ------ //
        for(int j = 1; j < width-1; j++) { // ---- x direction --- //

            p  = i * width + j;
            p1 = p - 1;
            p2 = p - width;

            v1x = flow_uv[0].at<float>(i,j) - flow_uv[0].at<float>(i,j-1);
            v2y = flow_uv[1].at<float>(i,j) - flow_uv[1].at<float>(i-1,j);
            divergence.at<float>(i,j) = v1x + v2y;
        }
    }

    // ----- up down line border area ------ //
    for (int j = 1; j < width-1; j++) {
        divergence.at<float>(0,j) =  flow_uv[0].at<float>(0,j) -  flow_uv[0].at<float>(0,j-1) + flow_uv[1].at<float>(0,j);
        divergence.at<float>(height-1,j) =  flow_uv[0].at<float>(height-1,j) -  flow_uv[0].at<float>(height-1,j-1) - flow_uv[1].at<float>(height-1,j);;
    }

    // ----- left right line border area ------ //
    for (int i = 1; i < height-1; i++) {
        divergence.at<float>(i,0) =  flow_uv[0].at<float>(i,0) +  flow_uv[1].at<float>(i,0) - flow_uv[1].at<float>(i-1,0);
        divergence.at<float>(i,width-1) =  -flow_uv[0].at<float>(i,width-1) +  flow_uv[1].at<float>(i,width-1) - flow_uv[1].at<float>(i-1,width-1);
    }
    // ---------------- four quarter points ------------ //
    divergence.at<float>(0,0) = flow_uv[0].at<float>(0,0) + flow_uv[1].at<float>(0,0);
    divergence.at<float>(0,width -1) = flow_uv[1].at<float>(0,width-1) - flow_uv[0].at<float>(0,width-2);
    divergence.at<float>(height -1,0) = flow_uv[0].at<float>(height-1,0) - flow_uv[1].at<float>(height-2,0);
    divergence.at<float>(height -1,width -1) = -flow_uv[0].at<float>(height-1,width-2) + flow_uv[0].at<float>(height -1,width-1);

    return divergence;
}
cv::Mat distance_mask(cv::Mat rgbImage,cv::Mat rgbImageW_seq,cv::Mat divergence,int width,int height,float mask_sigma) {
    std::vector<cv::Mat> rgbVects(3), rgbImage_W_Vects(3);
    cv::split(rgbImage, rgbVects);
    cv::split(rgbImageW_seq, rgbImage_W_Vects);
    cv::Mat maskdist;
    maskdist.create(height, width, CV_32FC1);
    for (int j = 0; j < height; j++) {
        //   std::cout<<"j = ------------- "<<j<<std::endl;
        for (int i = 0; i < width; i++) {
            //  --------- Color Term --------- //
            float fDif = 0.0f;
            float fDist = 0.0f;
            for (int ii = 0; ii < 3; ii++) {
                fDif = float(rgbVects[ii].at<uchar>(j, i)) - float(rgbImage_W_Vects[ii].at<uchar>(j, i));
                fDist += fDif;
            }
            fDist /= 3.0;
            maskdist.at<float>(j, i) = expf(-fDif / mask_sigma);

            // --------- divergence term ------ //
            float divV = divergence.at<float>(j, i);
            if (divV < 0)
                maskdist.at<float>(j, i) *= exp(divV);
            // ---------- mask binarization ------- //
            maskdist.at<float>(j, i)  = maskdist.at<float>(j, i) *= exp(divV)>hBin?1.0f:0.0f;

        }
    }
    return maskdist;
}
cv::Mat CheckLeftRightFlow(cv::Mat flow,cv::Mat flow_inv,int width,int height,float lrflowTh){
    cv::Mat flow_uv[2],flow_inv_uv[2];
    cv::split(flow, flow_uv);
    cv::split(flow_inv, flow_inv_uv);
    cv::Mat masklr;
    masklr.create(height,width,CV_32FC1);
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int OFxRev = flow_uv[0].at<float>(j,i)+(int)rintf(flow_uv[0].at<float>(j,i));
            int OFyRev = flow_uv[1].at<float>(j,i)+(int)rintf(flow_uv[1].at<float>(j,i));
            if (OFxRev >= 0 && OFxRev < width && OFyRev >= 0 && OFyRev < height) {
                float sum1 = fabsf(flow_uv[0].at<float>(j,i)+flow_inv_uv[0].at<float>(OFyRev,OFxRev));
                float sum2 = fabsf(flow_uv[1].at<float>(j,i)+flow_inv_uv[1].at<float>(OFyRev,OFxRev));
                float sum = MAX(sum1,sum2);
                masklr.at<float>(j,i) = sum>lrflowTh?0.0f:1.0f;
            }
            else
                masklr.at<float>(j,i) = 0.0f;
        }
    }
    return masklr;
}
cv::Mat InterSect(cv::Mat maskdist,cv::Mat masklr,int width,int height){
    cv::Mat maskout;
    maskout.create(height,width,CV_32FC1);
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            maskout.at<float>(j,i) = maskdist.at<float>(j,i)*masklr.at<float>(j,i);
        }
    }
    return maskout;
}