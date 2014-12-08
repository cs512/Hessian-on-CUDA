/*
 * deviceHelper.h
 *
 *  Created on: 2014-12-8
 *      Author: wangjz
 */

#ifndef DEVICEHELPER_H_
#define DEVICEHELPER_H_

#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
using namespace cv::gpu;

GpuMat cuHalfImage(const GpuMat &input);
GpuMat cuDoubleImage(const GpuMat &input);

#endif /* DEVICEHELPER_H_ */
