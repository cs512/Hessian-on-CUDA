/*
 * pyramid.cu
 *
 *  Created on: 2014-12-4
 *      Author: wangjz
 */

#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <thrust/device_vector.h>
#include "pyramid.h"
#include "hostHelpers.h"
#include "deviceHelpers.h"
#include "texture_binder.hpp"

#include "hesaff/helpers.h"

using namespace cv;
using namespace cv::gpu;

texture<float, 2, cudaReadModeElementType> texRef;
Stream stream1;

int CUHessianDetector::getHessianPointType(float *ptr, float value)
{
    if (value < 0)
        return CUHessianDetector::HESSIAN_SADDLE;
    else
    {
        // at this point we know that 2x2 determinant is positive
        // so only check the remaining 1x1 subdeterminant
        float Lxx = (ptr[-1]-2*ptr[0]+ptr[1]);
        if (Lxx < 0)
            return CUHessianDetector::HESSIAN_DARK;
        else
            return CUHessianDetector::HESSIAN_BRIGHT;
    }
}

__global__ void performHessianResponse(gpu::PtrStepSz<float> out, float norm2)
{
    float v11, v12, v13, v21, v22, v23, v31, v32, v33;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x > out.cols - 3) || (y > out.rows - 3))
        return;
    //int offset = x + y * cols;
    /* fill in shift registers at the beginning of the row */
    /* fetch remaining values (last column) */
    v11 = tex2D(texRef, x, y);     v12 = tex2D(texRef, x, y + 1);     v13 = tex2D(texRef, x, y + 2);
    v21 = tex2D(texRef, x + 1, y); v22 = tex2D(texRef, x + 1, y + 1); v23 = tex2D(texRef, x + 1, y + 2);
    v31 = tex2D(texRef, x + 2, y); v32 = tex2D(texRef, x + 2, y + 1); v33 = tex2D(texRef, x + 2, y + 2);
    // compute 3x3 Hessian values from symmetric differences.
    float Lxx = (v21 - 2 * v22 + v23);
    float Lyy = (v12 - 2 * v22 + v32);
    float Lxy = (v13 - v11 + v31 - v33) * 0.25f;

    /* normalize and write out */
    out(y + 1, x + 1) = (Lxx * Lyy - Lxy * Lxy) * norm2;
}

void performHessianResponseCaller(const PtrStepSz<float>& inputImage, PtrStepSz<float> outputImage, float norm2)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    // TODO handle error
    dim3 blocks((15 - 2 + cols) / 16, (15 - 2 + rows) / 16);
    dim3 threads(16, 16);
    TextureBinder tb(inputImage, texRef);
//    performHessianResponse<<<blocks, threads>>>(inputImage, outputImage, norm2);
    performHessianResponse<<<blocks, threads>>>(outputImage, norm2);
}

GpuMat CUHessianDetector::hessianResponse(const gpu::GpuMat &inputImage, float norm)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    CV_Assert(inputImage.type() == CV_32FC1);
    float norm2 = norm * norm;
    gpu::GpuMat outputImage(rows, cols, CV_32FC1);
    outputImage.setTo(Scalar::all(0));
    performHessianResponseCaller(inputImage, outputImage, norm2);
    return outputImage;
}

__device__ bool isMax(float val, const PtrStep<float> pix, int row, int col)
{
   for (int r = row - 1; r <= row + 1; r++)
   {
      for (int c = col - 1; c <= col + 1; c++)
         if (pix(r, c) > val)
            return false;
   }
   return true;
}

__device__ bool isMin(float val, const PtrStep<float> pix, int row, int col)
{
   for (int r = row - 1; r <= row + 1; r++)
   {
      for (int c = col - 1; c <= col + 1; c++)
         if (pix(r, c) < val)
            return false;
   }
   return true;
}

__global__ void performFindLevelKeypointsThreshold(const float border, const float positiveThreshold,
        const float negativeThreshold, const PtrStepSz<float> low, const PtrStepSz<float> cur,
        const PtrStepSz<float> high, PtrStep<unsigned char> out)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if ((col >= cur.cols - border) || (col < border))
        return;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if ((row >= cur.rows - border) || (row < border))
        return;
    const float val = cur(row,col);
    if ( (val > positiveThreshold && (isMax(val, cur, row, col) && isMax(val, low, row, col) && isMax(val, high, row, col))) ||
         (val < negativeThreshold && (isMin(val, cur, row, col) && isMin(val, low, row, col) && isMin(val, high, row, col))) )
    {
        out(row, col) = 1;
    }
    return;
}

//if ( (val > positiveThreshold && (isMax(val, cur, row, col) && isMax(val, low, row, col) && isMax(val, high, row, col))) ||
//     (val < negativeThreshold && (isMin(val, cur, row, col) && isMin(val, low, row, col) && isMin(val, high, row, col))) )
// // either positive -> local max. or negative -> local min.
//    //localizeKeypoint(row, col, curScale, pixelDistance);

void CUHessianDetector::findLevelKeypoints(float curScale, float pixelDistance)
{
    assert(par.border >= 2);
    const int rows = cur.rows;
    const int cols = cur.cols;
    // TODO handle error
    dim3 blocks((15 - 2 + cols) / 16, (15 - 2 + rows) / 16);
    dim3 threads(16, 16);
    GpuMat tempMap = GpuMat(cur.rows, cur.cols, CV_8UC1);
    tempMap.setTo(Scalar::all(0));
//    performFindLevelKeypoints<<<blocks, threads>>>(par.border, curScale, pixelDistance,
//            positiveThreshold, negativeThreshold, low, cur, high);
    performFindLevelKeypointsThreshold<<<blocks, threads>>>(par.border, positiveThreshold, negativeThreshold,
        low, cur, high, tempMap);
    stream1.waitForCompletion();
#ifdef DEBUG_H_PK
          tempMap.download((this->results.back()));
#endif
    return;
}


void CUHessianDetector::detectOctaveKeypoints(const GpuMat &firstLevel, float pixelDistance, GpuMat &nextOctaveFirstLevel)
{
    cOctaveMap = Mat(firstLevel.rows, firstLevel.cols, CV_8UC1, Scalar(0));
    float sigmaStep = pow(2.0f, 1.0f / (float) par.numberOfScales);
    float curSigma = par.initialSigma;
    blur = firstLevel;
    cBlur = Mat(blur.rows, blur.cols, blur.type());
    stream1.enqueueDownload(blur, cBlur);
//    blur.download(cBlur, stream1);

    cur = hessianResponse(blur, curSigma*curSigma);
//    cur.download(cCur, stream1);
    cCur = Mat(cur.rows, cur.cols, cur.type());
    stream1.enqueueDownload(cur, cCur);
    int numLevels = 1;

    for (int i = 1; i < par.numberOfScales+2; i++)
    {
        // compute the increase necessary for the next level and compute the next level
        float sigma = curSigma * sqrt(sigmaStep * sigmaStep - 1.0f);
        // do the blurring
        GpuMat nextBlur = cuGaussianBlur(blur, sigma);
        Mat cNextBlur= Mat(nextBlur.rows, nextBlur.cols, nextBlur.type());
//        nextBlur.download(cNextBlur, stream1);
        stream1.enqueueDownload(nextBlur, cNextBlur);
        // the next level sigma
        sigma = curSigma*sigmaStep;
        // compute response for current level
        high = hessianResponse(nextBlur, sigma*sigma);
//        high.download(cHigh, stream1);
        cHigh = Mat(high.rows, high.cols, high.type());
        stream1.enqueueDownload(high, cHigh);
        numLevels ++;
        // if we have three consecutive responses
        if (numLevels == 3)
        {
            // find keypoints in this part of octave for curLevel
#ifdef DEBUG_H_PK
          this->results.push_back(Mat(cur.rows, cur.cols, CV_8UC1, Scalar(0)));
#endif
            findLevelKeypoints(curSigma, pixelDistance);
            numLevels--;
        }
        if (i == par.numberOfScales)
            // downsample the right level for the next octave
            nextOctaveFirstLevel = cuHalfImage(nextBlur);
//        stream1.waitForCompletion();
        prevBlur = blur; blur = nextBlur;
        cPrevBlur = cBlur; cBlur = cNextBlur;
        // shift to the next response
        low = cur; cur = high;
        cLow = cCur; cCur = cHigh;
        curSigma *= sigmaStep;
    }
}

void CUHessianDetector::detectPyramidKeypoints(const GpuMat &image)
{
   float curSigma = 0.5f;
   float pixelDistance = 1.0f;
   GpuMat   firstLevel;

   if (par.upscaleInputImage > 0)
   {
      firstLevel = cuDoubleImage(image);
      pixelDistance *= 0.5f;
      curSigma *= 2.0f;
   } else
      firstLevel = image.clone();

   // prepare first octave input image
   if (par.initialSigma > curSigma)
   {
      float sigma = sqrt(par.initialSigma * par.initialSigma - curSigma * curSigma);
      cuGaussianBlurInplace(firstLevel, sigma);
   }

   // while there is sufficient size of image
   int minSize = 2 * par.border + 2;
   while (firstLevel.rows > minSize && firstLevel.cols > minSize)
   {
      GpuMat nextOctaveFirstLevel;
      detectOctaveKeypoints(firstLevel, pixelDistance, nextOctaveFirstLevel);
      pixelDistance *= 2.0;
      // firstLevel gets destroyed in the process
      firstLevel = nextOctaveFirstLevel;
   }
}

// it seems 0.6 works better than 0.5 (as in DL paper)
#define MAX_SUBPIXEL_SHIFT 0.6

// we don't care about border effects
#define POINT_SAFETY_BORDER  3

void CUHessianDetector::localizeKeypoint(int r, int c, float curScale, float pixelDistance)
{
    const int cols = cur.cols;
    const int rows = cur.rows;

    float b[3] = {};
    float val = 0;
    bool converged = false;
    int nr = r, nc = c;

    for (int iter=0; iter<5; iter++)
    {
       // take current position
       r = nr; c = nc;

       float dxx = cCur.at<float>(r,c-1) - 2.0f * cCur.at<float>(r,c) + cCur.at<float>(r,c+1);
       float dyy = cCur.at<float>(r-1,c) - 2.0f * cCur.at<float>(r,c) + cCur.at<float>(r+1,c);
       float dss = cLow.at<float>(r,c  ) - 2.0f * cCur.at<float>(r,c) + cHigh.at<float>(r, c);

       float dxy = 0.25f*(cCur.at<float>(r+1,c+1) - cCur.at<float>(r+1,c-1) - cCur.at<float>(r-1,c+1) + cCur.at<float>(r-1,c-1));
       // check edge like shape of the response function in first iteration
       if (0 == iter)
       {
          float edgeScore = (dxx + dyy)*(dxx + dyy)/(dxx * dyy - dxy * dxy);
          if (edgeScore >= edgeScoreThreshold || edgeScore < 0)
             // local neighbourhood looks like an edge
             return;
       }
       float dxs = 0.25f*(cHigh.at<float>(r  ,c+1) - cHigh.at<float>(r  ,c-1) - cLow.at<float>(r  ,c+1) + cLow.at<float>(r  ,c-1));
       float dys = 0.25f*(cHigh.at<float>(r+1,c  ) - cHigh.at<float>(r-1,c  ) - cLow.at<float>(r+1,c  ) + cLow.at<float>(r-1,c  ));

       float A[9];
       A[0] = dxx; A[1] = dxy; A[2] = dxs;
       A[3] = dxy; A[4] = dyy; A[5] = dys;
       A[6] = dxs; A[7] = dys; A[8] = dss;

       float dx = 0.5f*(cCur.at<float>(r,c+1) - cCur.at<float>(r,c-1));
       float dy = 0.5f*(cCur.at<float>(r+1,c) - cCur.at<float>(r-1,c));
       float ds = 0.5f*(cHigh.at<float>(r,c)  - cLow.at<float>(r,c));

       b[0] = - dx; b[1] = - dy; b[2] = - ds;

       solveLinear3x3(A, b);

       // check if the solution is valid
       if (isnan(b[0]) || isnan(b[1]) || isnan(b[2]))
          return;

       // aproximate peak value
       val = cCur.at<float>(r,c) + 0.5f * (dx*b[0] + dy*b[1] + ds*b[2]);

       // if we are off by more than MAX_SUBPIXEL_SHIFT, update the position and iterate again
       if (b[0] >  MAX_SUBPIXEL_SHIFT) { if (c < cols - POINT_SAFETY_BORDER) nc++; else return; }
       if (b[1] >  MAX_SUBPIXEL_SHIFT) { if (r < rows - POINT_SAFETY_BORDER) nr++; else return; }
       if (b[0] < -MAX_SUBPIXEL_SHIFT) { if (c >        POINT_SAFETY_BORDER) nc--; else return; }
       if (b[1] < -MAX_SUBPIXEL_SHIFT) { if (r >        POINT_SAFETY_BORDER) nr--; else return; }

       if (nr == r && nc == c)
       {
          // converged, displacement is sufficiently small, terminate here
          // TODO: decide if we want only converged local extrema...
          converged = true;
          break;
       }
    }

    // if spatial localization was all right and the scale is close enough...
    if (fabs(b[0]) > 1.5 || fabs(b[1]) > 1.5 || fabs(b[2]) > 1.5 || fabs(val) < finalThreshold || cOctaveMap.at<unsigned char>(r,c) > 0)
       return;

    // mark we were here already
    cOctaveMap.at<unsigned char>(r,c) = 1;

    // output keypoint
    float scale = curScale * pow(2.0f, b[2] / par.numberOfScales );

    // set point type according to final location
    int type = getHessianPointType(blur.ptr<float>(r)+c, val);

    // point is now scale and translation invariant, add it...
    if (hessianKeypointCallback)
       hessianKeypointCallback->onHessianKeypointDetected(cPrevBlur, pixelDistance*(c + b[0]), pixelDistance*(r + b[1]), pixelDistance*scale, pixelDistance, type, val);
}
