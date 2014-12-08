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
using namespace cv;

class CUHessianKeypointCallback
{
public:
   virtual void onHessianKeypointDetected(const Mat &blur, float x, float y, float s, float pixelDistance, int type, float response) = 0;
};

class CUHessianDetector {
public:
    CUHessianKeypointCallback *hessianKeypointCallback;
    gpu::GpuMat hessianResponse(const gpu::GpuMat &inputImage, float norm);
    void detectPyramidKeypoints(const Mat &image);
    void detectOctaveKeypoints(const Mat &firstLevel, float pixelDistance, Mat &nextOctaveFirstLevel);
    void findLevelKeypoints(float curScale, float pixelDistance);
    void localizeKeypoint(int r, int c, float curScale, float pixelDistance);
private:
    gpu::GpuMat octaveMap;
    gpu::GpuMat prevBlur, blur;
    gpu::GpuMat low, cur, high;
};



#endif /* PYRAMID_H_ */
