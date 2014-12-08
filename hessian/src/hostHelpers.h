/*
 * hostHelper.h
 *
 *  Created on: 2014-12-8
 *      Author: wangjz
 */

#ifndef HOSTHELPER_H_
#define HOSTHELPER_H_

#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
//using namespace cv;
using namespace cv::gpu;

void cuGaussianBlurInplace(GpuMat &inplace, float sigma);
GpuMat cuGussianBlur(const GpuMat input, float sigma);


#endif /* HOSTHELPER_H_ */
