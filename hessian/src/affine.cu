/*
 * affine.cu
 *
 *  Created on: 2014-12-20
 *      Author: wangjz
 */
#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include "affine.h"
#include "hostHelpers.h"
#include "deviceHelpers.h"
//#include "deviceHelpers.cuh"
#include "texture_binder.hpp"

using namespace cv;
using namespace cv::gpu;

__global__ void performComputeGradient(const PtrStepSz<float> img, PtrStep<float> gradx, PtrStep<float> grady)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= img.cols)
        return;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y >= img.rows)
        return;
    float xgrad, ygrad;
    if (x == 0)
        xgrad = img(y,x+1) - img(y,x);
    else if (x == img.cols-1)
        xgrad = img(y,x) - img(y,x-1);
    else
        xgrad = img(y,x+1) - img(y,x-1);

    if (y == 0)
        ygrad = img(y+1,x) - img(y,x);
    else if (y == img.rows-1)
        ygrad = img(y,x) - img(y-1,x);
    else
        ygrad = img(y+1,x) - img(y-1,x);
    gradx(y,x) = xgrad;
    grady(y,x) = ygrad;
    return;
}

void cuComputeGradient(const GpuMat &img, GpuMat &gradx, GpuMat &grady)
{
    const int width = img.cols;
    const int height = img.rows;
    dim3 blocks((15 + width) / 16, (15 + height) / 16);
    dim3 threads(16, 16);
    performComputeGradient<<<blocks, threads>>>(img, gradx, grady);
}

__global__ void performEstimateSMM(float *ptrA, float *ptrB, float* ptrC, const PtrStepSz<float> mask,
        PtrStep<float> fx, PtrStep<float> fy)
{
    __shared__ float shareA, shareB, shareC;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (threadIdx.x == 0)
    {
        shareA = 0;
        shareB = 0;
        shareC = 0;
    }
    if(x + y == 0)
    {
        *ptrA = 0;
        *ptrB = 0;
        *ptrC = 0;
    }
    __syncthreads();
    if ((x < mask.cols) && (y < mask.rows))
    {
        const float v = mask(y, x);
        const float gxx = fx(y, x);
        const float gyy = fy(y, x);
        const float gxy = gxx * gyy;

        atomicAdd(&shareA, gxx * gxx * v);
//        a += gxx * gxx * v;
        atomicAdd(&shareB, gxy * v);
//        b += gxy * v;
        atomicAdd(&shareC, gyy * gyy * v);
//        c += gyy * gyy * v;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(ptrA, shareA);
        atomicAdd(ptrB, shareB);
        atomicAdd(ptrC, shareC);
//        shareA = 0;
//        shareB = 0;
//        shareC = 0;
    }

    return;
}

bool CUAffineShape::findAffineShape(const GpuMat &blur, float x, float y, float s, float pixelDistance, int type, float response)
{
    float eigen_ratio_act = 0.0f, eigen_ratio_bef = 0.0f;
    float u11 = 1.0f, u12 = 0.0f, u21 = 0.0f, u22 = 1.0f, l1 = 1.0f, l2 = 1.0f;
    float lx = x/pixelDistance, ly = y/pixelDistance;
    float ratio = s/(par.initialSigma*pixelDistance);
    // kernel size...
    const int maskPixels = par.smmWindowSize * par.smmWindowSize;

    for (int l = 0; l < par.maxIterations; l ++)
    {
        // warp input according to current shape matrix
        cuInterpolate(blur, lx, ly, u11*ratio, u12*ratio, u21*ratio, u22*ratio, img);

        // compute SMM on the warped patch
        float a = 0, b = 0, c = 0;
        float *maskptr = mask.ptr<float>(0);
        float *pfx = fx.ptr<float>(0), *pfy = fy.ptr<float>(0);

        cuComputeGradient(img, fx, fy);
        // estimate SMM
        dim3 blocks((15 + mask.cols) / 16, (15 + mask.rows) / 16);
        dim3 threads(16, 16);
        float *devPtrA = NULL;
        float *devPtrB = NULL;
        float *devPtrC = NULL;
        cudaMalloc((void **)&devPtrA, sizeof(float));
        cudaMalloc((void **)&devPtrB, sizeof(float));
        cudaMalloc((void **)&devPtrC, sizeof(float));
        performEstimateSMM<<<blocks, threads>>>(devPtrA, devPtrB, devPtrC, mask, fx, fy);
        cudaMemcpy(&a, devPtrA, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b, devPtrB, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&c, devPtrC, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(devPtrA);
        cudaFree(devPtrB);
        cudaFree(devPtrC);
//        for (int i = 0; i < maskPixels; ++i)
//        {
//            const float v = (*maskptr);
//            const float gxx = *pfx;
//            const float gyy = *pfy;
//            const float gxy = gxx * gyy;
//
//            a += gxx * gxx * v;
//            b += gxy * v;
//            c += gyy * gyy * v;
//            pfx++; pfy++; maskptr++;
//        }
        a /= maskPixels; b /= maskPixels; c /= maskPixels;

        // compute inverse sqrt of the SMM
        invSqrt(a, b, c, l1, l2);

        // update eigen ratios
        eigen_ratio_bef = eigen_ratio_act;
        eigen_ratio_act = 1 - l2 / l1;

        // accumulate the affine shape matrix
        float u11t = u11, u12t = u12;

        u11 = a*u11t+b*u21; u12 = a*u12t+b*u22;
        u21 = b*u11t+c*u21; u22 = b*u12t+c*u22;

        // compute the eigen values of the shape matrix
        if (!getEigenvalues(u11, u12, u21, u22, l1, l2))
         break;

        // leave on too high anisotropy
        if ((l1/l2>6) || (l2/l1>6))
         break;

        if (eigen_ratio_act < par.convergenceThreshold && eigen_ratio_bef < par.convergenceThreshold)
        {
            if (affineShapeCallback)
                affineShapeCallback->onAffineShapeFound(blur, x, y, s, pixelDistance, u11, u12, u21, u22, type, response, l);
            return true;
        }
    }
    return false;
}
