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
bool cuInterpolate(const GpuMat &im, const float ofsx, const float ofsy, const float a11,
        const float a12, const float a21, const float a22, GpuMat &res);

#ifdef __CUDACC__
template <typename ValueType> __device__
void swap(ValueType *a, ValueType *b);

__device__ void cuSolveLinear3x3(float *A, float *b);
#endif

#endif /* DEVICEHELPER_H_ */
