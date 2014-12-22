/*
 * pyramid.h
 *
 *  Created on: 2014-12-4
 *      Author: wangjz
 */

#ifndef PYRAMID_H_
#define PYRAMID_H_
#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <vector>
#include "debug.h"
using namespace cv;

class CUHessianKeypointCallback
{
public:
   virtual void onHessianKeypointDetected(const gpu::GpuMat &blur, float x, float y, float s, float pixelDistance, int type, float response) = 0;
};

struct CUPyramidParams
{
   // shall input image be upscaled ( > 0)
   int upscaleInputImage;
   // number of scale per octave
   int  numberOfScales;
   // amount of smoothing applied to the initial level of first octave
   float initialSigma;
   // noise dependent threshold on the response (sensitivity)
   float threshold;
   // ratio of the eigenvalues
   float edgeEigenValueRatio;
   // number of pixels ignored at the border of image
   int  border;
   CUPyramidParams()
    {
        upscaleInputImage = 0;
        numberOfScales = 3;
        initialSigma = 1.6f;
        threshold = 16.0f/3.0f; //0.04f * 256 / 3;
        edgeEigenValueRatio = 10.0f;
        border = 5;
    }
};

class CUHessianDetector {
    enum
    {
        HESSIAN_DARK   = 0,
        HESSIAN_BRIGHT = 1,
        HESSIAN_SADDLE = 2,
    };

public:
    CUHessianKeypointCallback *hessianKeypointCallback;
    CUPyramidParams par;
    CUHessianDetector(const CUPyramidParams &par) :
        edgeScoreThreshold((par.edgeEigenValueRatio + 1.0f)*(par.edgeEigenValueRatio + 1.0f)/par.edgeEigenValueRatio),
        finalThreshold(par.threshold * par.threshold),
        positiveThreshold(0.8 * finalThreshold),
        negativeThreshold(-positiveThreshold)
    {
        this->par = par;
        hessianKeypointCallback = 0;
    }
    gpu::GpuMat hessianResponse(const gpu::GpuMat &inputImage, float norm);

    void findLevelKeypoints(float curScale, float pixelDistance);
    void detectOctaveKeypoints(const gpu::GpuMat &firstLevel, float pixelDistance, gpu::GpuMat &nextOctaveFirstLevel);
    void detectPyramidKeypoints(const gpu::GpuMat &image);
    void detectPyramidKeypoints(const Mat &image);
    void localizeKeypoint(int r, int c, float curScale, float pixelDistance);
    int getHessianPointType(float *ptr, float value);
    void setHessianKeypointCallback(CUHessianKeypointCallback *callback)
          {
             hessianKeypointCallback = callback;
          }
#ifdef DEBUG_H_PK
    vector <Mat> results;
#endif
private:
    gpu::GpuMat prevBlur, blur;
    gpu::GpuMat low, cur, high;
    gpu::GpuMat octaveMap;
    const float edgeScoreThreshold;
    const float finalThreshold;
    const float positiveThreshold;
    const float negativeThreshold;
};



#endif /* PYRAMID_H_ */
