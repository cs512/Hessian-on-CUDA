/*
 * pyramid.cu
 *
 *  Created on: 2014-12-4
 *      Author: wangjz
 */

#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include "pyramid.h"

using namespace cv;
using namespace cv::gpu;

__global__ void performHessianResponse(const gpu::PtrStep<float> in, gpu::PtrStep<float> out, float norm,
        int cols, int rows)
{
    float v11, v12, v13, v21, v22, v23, v31, v32, v33;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x > cols - 3)
        return;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (y > rows - 3)
        return;
    //int offset = x + y * cols;
    /* fill in shift registers at the beginning of the row */
    /* fetch remaining values (last column) */
    v11 = in(y, x);     v12 = in(y + 1, x);     v13 = in(y + 2, x);
    v21 = in(y, x + 1); v22 = in(y + 1, x + 1); v23 = in(y + 2, x + 1);
    v31 = in(y, x + 2); v32 = in(y + 1, x + 2); v33 = in(y + 2, x + 2);
    // compute 3x3 Hessian values from symmetric differences.
    float Lxx = (v21 - 2 * v22 + v23);
    float Lyy = (v12 - 2 * v22 + v32);
    float Lxy = (v13 - v11 + v31 - v33) / 4.0f;

    /* normalize and write out */
    out(y + 1, x + 1) = (Lxx * Lyy - Lxy * Lxy) * norm;
}

void performHessianResponseCaller(const PtrStepSz<float>& inputImage, PtrStep<float> outputImage, float norm2)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    // TODO handle error
    dim3 blocks((15 - 2 + cols) / 16, (15 - 2 + rows) / 16);
    dim3 threads(16, 16);
    performHessianResponse<<<blocks, threads>>>(inputImage, outputImage, norm2, cols, rows);
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

/*
GpuMat CUHessianDetector::hessianResponse(const gpu::GpuMat &inputImage, float norm)
{
    const int rows = inputImage.rows;
    const int cols = inputImage.cols;
    CV_Assert(inputImage.type() == CV_32FC1);
    float norm2 = norm * norm;
    gpu::GpuMat outputImage(rows, cols, CV_32FC1);
    performHessianResponseCaller((PtrStepSz<float>)inputImage, (PtrStepSz<float>)outputImage, norm2);
    return outputImage;
}
*/
/*
void CUHessianDetector::detectPyramidKeypoints(const Mat &image)
{

    float curSigma = 0.5f;
    float pixelDistance = 1.0f;
    Mat firstLevel;

    if (par.upscaleInputImage > 0)
    {
        // TODO CUDAble
        firstLevel = doubleImage(image);
        pixelDistance *= 0.5f;
        curSigma *= 2.0f;
    }
    else
        firstLevel = image.clone();

    // prepare first octave input image
    if (par.initialSigma > curSigma)
    {
        float sigma = sqrt(
                par.initialSigma * par.initialSigma - curSigma * curSigma);
        // CV
        gaussianBlurInplace(firstLevel, sigma);
    }

    // while there is sufficient size of image
    int minSize = 2 * par.border + 2;
    while (firstLevel.rows > minSize && firstLevel.cols > minSize)
    {
        Mat nextOctaveFirstLevel;
        // TODO CUDAble
        detectOctaveKeypoints(firstLevel, pixelDistance, nextOctaveFirstLevel);
        pixelDistance *= 2.0;
        // firstLevel gets destroyed in the process
        firstLevel = nextOctaveFirstLevel;
    }
}

void CUHessianDetector::detectOctaveKeypoints(const Mat &firstLevel, float pixelDistance, Mat &nextOctaveFirstLevel)
{
    octaveMap = Mat::zeros(firstLevel.rows, firstLevel.cols, CV_8UC1);
    float sigmaStep = pow(2.0f, 1.0f / (float) par.numberOfScales);
    float curSigma = par.initialSigma;
    blur = firstLevel;
    cur = hessianResponse(blur, curSigma*curSigma);
    int numLevels = 1;

    for (int i = 1; i < par.numberOfScales+2; i++)
    {
       // compute the increase necessary for the next level and compute the next level
       float sigma = curSigma * sqrt(sigmaStep * sigmaStep - 1.0f);
       // do the blurring
       Mat nextBlur = gaussianBlur(blur, sigma);
       // the next level sigma
       sigma = curSigma*sigmaStep;
       // compute response for current level
       high = hessianResponse(nextBlur, sigma*sigma);
       numLevels ++;
       // if we have three consecutive responses
       if (numLevels == 3)
       {
          // find keypoints in this part of octave for curLevel
          findLevelKeypoints(curSigma, pixelDistance);
          numLevels--;
       }
       if (i == par.numberOfScales)
          // downsample the right level for the next octave
          nextOctaveFirstLevel = halfImage(nextBlur);
       prevBlur = blur; blur = nextBlur;
       // shift to the next response
       low = cur; cur = high;
       curSigma *= sigmaStep;
    }
}

void CUHessianDetector::findLevelKeypoints(float curScale, float pixelDistance)
{
    assert(par.border >= 2);
    const int rows = cur.rows;
    const int cols = cur.cols;
    for (int r = par.border; r < (rows - par.border); r++)
    {
       for (int c = par.border; c < (cols - par.border); c++)
       {
          const float val = cur.at<float>(r,c);
          if ( (val > positiveThreshold && (isMax(val, cur, r, c) && isMax(val, low, r, c) && isMax(val, high, r, c))) ||
               (val < negativeThreshold && (isMin(val, cur, r, c) && isMin(val, low, r, c) && isMin(val, high, r, c))) )
             // either positive -> local max. or negative -> local min.
             localizeKeypoint(r, c, curScale, pixelDistance);
       }
    }
}

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

       float dxx = cur.at<float>(r,c-1) - 2.0f * cur.at<float>(r,c) + cur.at<float>(r,c+1);
       float dyy = cur.at<float>(r-1,c) - 2.0f * cur.at<float>(r,c) + cur.at<float>(r+1,c);
       float dss = low.at<float>(r,c  ) - 2.0f * cur.at<float>(r,c) + high.at<float>(r, c);

       float dxy = 0.25f*(cur.at<float>(r+1,c+1) - cur.at<float>(r+1,c-1) - cur.at<float>(r-1,c+1) + cur.at<float>(r-1,c-1));
       // check edge like shape of the response function in first iteration
       if (0 == iter)
       {
          float edgeScore = (dxx + dyy)*(dxx + dyy)/(dxx * dyy - dxy * dxy);
          if (edgeScore >= edgeScoreThreshold || edgeScore < 0)
             // local neighbourhood looks like an edge
             return;
       }
       float dxs = 0.25f*(high.at<float>(r  ,c+1) - high.at<float>(r  ,c-1) - low.at<float>(r  ,c+1) + low.at<float>(r  ,c-1));
       float dys = 0.25f*(high.at<float>(r+1,c  ) - high.at<float>(r-1,c  ) - low.at<float>(r+1,c  ) + low.at<float>(r-1,c  ));

       float A[9];
       A[0] = dxx; A[1] = dxy; A[2] = dxs;
       A[3] = dxy; A[4] = dyy; A[5] = dys;
       A[6] = dxs; A[7] = dys; A[8] = dss;

       float dx = 0.5f*(cur.at<float>(r,c+1) - cur.at<float>(r,c-1));
       float dy = 0.5f*(cur.at<float>(r+1,c) - cur.at<float>(r-1,c));
       float ds = 0.5f*(high.at<float>(r,c)  - low.at<float>(r,c));

       b[0] = - dx; b[1] = - dy; b[2] = - ds;

       solveLinear3x3(A, b);

       // check if the solution is valid
       if (isnan(b[0]) || isnan(b[1]) || isnan(b[2]))
          return;

       // aproximate peak value
       val = cur.at<float>(r,c) + 0.5f * (dx*b[0] + dy*b[1] + ds*b[2]);

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
    if (fabs(b[0]) > 1.5 || fabs(b[1]) > 1.5 || fabs(b[2]) > 1.5 || fabs(val) < finalThreshold || octaveMap.at<unsigned char>(r,c) > 0)
       return;

    // mark we were here already
    octaveMap.at<unsigned char>(r,c) = 1;

    // output keypoint
    float scale = curScale * pow(2.0f, b[2] / par.numberOfScales );

    // set point type according to final location
    int type = getHessianPointType(blur.ptr<float>(r)+c, val);

    // point is now scale and translation invariant, add it...
    if (hessianKeypointCallback)
       hessianKeypointCallback->onHessianKeypointDetected(prevBlur, pixelDistance*(c + b[0]), pixelDistance*(r + b[1]), pixelDistance*scale, pixelDistance, type, val);
}
*/
