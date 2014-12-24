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
/*
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

bool CUAffineShape::findAffineShape(const Mat &blur, float x, float y, float s, float pixelDistance, int type, float response)
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
*/

void computeGradient(const Mat &img, Mat &gradx, Mat &grady);
//{
//   const int width = img.cols;
//   const int height = img.rows;
//   for (int r = 0; r < height; ++r)
//      for (int c = 0; c < width; ++c)
//      {
//         float xgrad, ygrad;
//         if (c == 0) xgrad = img.at<float>(r,c+1) - img.at<float>(r,c); else
//            if (c == width-1) xgrad = img.at<float>(r,c) - img.at<float>(r,c-1); else
//               xgrad = img.at<float>(r,c+1) - img.at<float>(r,c-1);
//
//         if (r == 0) ygrad = img.at<float>(r+1,c) - img.at<float>(r,c); else
//            if (r == height-1) ygrad = img.at<float>(r,c) - img.at<float>(r-1,c); else
//               ygrad = img.at<float>(r+1,c) - img.at<float>(r-1,c);
//
//         gradx.at<float>(r,c) = xgrad;
//         grady.at<float>(r,c) = ygrad;
//      }
//}

bool CUAffineShape::findAffineShape(const Mat &blur, float x, float y, float s, float pixelDistance, int type, float response)
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
      interpolate(blur, lx, ly, u11*ratio, u12*ratio, u21*ratio, u22*ratio, img);

      // compute SMM on the warped patch
      float a = 0, b = 0, c = 0;
      float *maskptr = mask.ptr<float>(0);
      float *pfx = fx.ptr<float>(0), *pfy = fy.ptr<float>(0);

      computeGradient(img, fx, fy);

      // estimate SMM
      for (int i = 0; i < maskPixels; ++i)
      {
         const float v = (*maskptr);
         const float gxx = *pfx;
         const float gyy = *pfy;
         const float gxy = gxx * gyy;

         a += gxx * gxx * v;
         b += gxy * v;
         c += gyy * gyy * v;
         pfx++; pfy++; maskptr++;
      }
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

bool CUAffineShape::normalizeAffine(const Mat &img, float x, float y, float s, float a11, float a12, float a21, float a22)
{
   // determinant == 1 assumed (i.e. isotropic scaling should be separated in mrScale
   assert( fabs(a11*a22-a12*a21 - 1.0f) < 0.01);
   float mrScale = ceil(s * par.mrSize); // half patch size in pixels of image

   int   patchImageSize = 2*int(mrScale)+1; // odd size
   float imageToPatchScale = float(patchImageSize) / float(par.patchSize);  // patch size in the image / patch size -> amount of down/up sampling

   // is patch touching boundary? if yes, ignore this feature
   if (interpolateCheckBorders(img, x, y, a11*imageToPatchScale, a12*imageToPatchScale, a21*imageToPatchScale, a22*imageToPatchScale, patch))
      return true;

   if (imageToPatchScale > 0.4)
   {
      // the pixels in the image are 0.4 apart + the affine deformation
      // leave +1 border for the bilinear interpolation
      patchImageSize += 2;
      size_t wss = patchImageSize*patchImageSize*sizeof(float);
      if (wss >= workspace.size())
         workspace.resize(wss);

      Mat smoothed(patchImageSize, patchImageSize, CV_32FC1, (void *)&workspace.front());
      // interpolate with det == 1
      if (!interpolate(img, x, y, a11, a12, a21, a22, smoothed))
      {
         // smooth accordingly
         gaussianBlurInplace(smoothed, 1.5f*imageToPatchScale);
         // subsample with corresponding scale
         bool touchesBoundary = interpolate(smoothed, (float)(patchImageSize>>1), (float)(patchImageSize>>1), imageToPatchScale, 0, 0, imageToPatchScale, patch);
         assert(!touchesBoundary);
      } else
         return true;
   } else {
      // if imageToPatchScale is small (i.e. lot of oversampling), affine normalize without smoothing
      a11 *= imageToPatchScale; a12 *= imageToPatchScale;
      a21 *= imageToPatchScale; a22 *= imageToPatchScale;
      // ok, do the interpolation
      bool touchesBoundary = interpolate(img, x, y, a11, a12, a21, a22, patch);
      assert(!touchesBoundary);
   }
   return false;
}
