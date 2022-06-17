//
// Created by hong on 22-5-14.
//

#include "display_flow.h"

cv::Mat flow_rgb_cal(cv::Mat flow){
    cv::Mat flow_uv[2];
    cv::Mat mag, ang;
    cv::Mat hsv_split[3], hsv;
    cv::Mat rgb;
    cv::split(flow, flow_uv);
    multiply(flow_uv[1], -1, flow_uv[1]);
    cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    hsv_split[0] = ang;
    hsv_split[1] = mag;
    hsv_split[2] = Mat::ones(ang.size(), ang.type());
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    return rgb;
}
