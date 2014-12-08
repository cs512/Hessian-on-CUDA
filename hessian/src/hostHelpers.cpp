/*
 * hostHelper.cpp
 *
 *  Created on: 2014-12-8
 *      Author: wangjz
 */

#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include "hostHelpers.h"
using namespace cv;
using namespace cv::gpu;

GpuMat cuGaussianBlur(const GpuMat input, float sigma)
{
   GpuMat ret(input.rows, input.cols, input.type());
   int size = (int)(2.0 * 3.0 * sigma + 1.0); if (size % 2 == 0) size++;
   cv::gpu::GaussianBlur(input, ret, Size(size, size), sigma, sigma, BORDER_REPLICATE);
   return ret;
}

void cuGaussianBlurInplace(GpuMat &inplace, float sigma)
{
   int size = (int)(2.0 * 3.0 * sigma + 1.0); if (size % 2 == 0) size++;
   cv::gpu::GaussianBlur(inplace, inplace, Size(size, size), sigma, sigma, BORDER_REPLICATE);
}
