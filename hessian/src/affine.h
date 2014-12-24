/*
 * affine.h
 *
 *  Created on: 2014-12-20
 *      Author: wangjz
 */

#ifndef AFFINE_H_
#define AFFINE_H_

#include <vector>
//#include <cv.h>
#include <opencv2/gpu/gpumat.hpp>
#include <cv.h>
#include "hesaff/helpers.h"
using namespace cv::gpu;
using namespace cv;

void cuComputeGradient(const GpuMat &img, GpuMat &gradx, GpuMat &grady);

struct CUAffineShapeParams
{
   // number of affine shape interations
   int maxIterations;

   // convergence threshold, i.e. maximum deviation from isotropic shape at convergence
   float convergenceThreshold;

   // widht and height of the SMM mask
   int smmWindowSize;

   // width and height of the patch
   int patchSize;

   // amount of smoothing applied to the initial level of first octave
   float initialSigma;

   // size of the measurement region (as multiple of the feature scale)
   float mrSize;

   CUAffineShapeParams()
      {
         maxIterations = 16;
         initialSigma = 1.6f;
         convergenceThreshold = 0.05;
         patchSize = 41;
         smmWindowSize = 19;
         mrSize = 3.0f*sqrt(3.0f);
      }
};

struct CUAffineShapeCallback
{
   virtual void onAffineShapeFound(
      const Mat &blur,     // corresponding scale level
      float x, float y,     // subpixel, image coordinates
      float s,              // scale
      float pixelDistance,  // distance between pixels in provided blured image
      float a11, float a12, // affine shape matrix
      float a21, float a22,
      int type, float response, int iters) = 0;
};

struct CUAffineShape
{
public:
   CUAffineShape(const CUAffineShapeParams &par) :
      patch(par.patchSize, par.patchSize, CV_32FC1),
      mask(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
      img(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
      fx(par.smmWindowSize, par.smmWindowSize, CV_32FC1),
      fy(par.smmWindowSize, par.smmWindowSize, CV_32FC1)
      {
         this->par = par;
//         cv::Mat tempMask(par.smmWindowSize, par.smmWindowSize, CV_32FC1);
         computeGaussMask(mask);
//         mask.upload(tempMask);
         affineShapeCallback = 0;
         fx = cv::Scalar(0);
         fy = cv::Scalar(0);
      }

   ~CUAffineShape()
      {
      }

   // computes affine shape
   bool findAffineShape(const Mat &blur, float x, float y, float s, float pixelDistance, int type, float response);

   // fills patch with affine normalized neighbourhood around point in the img, enlarged mrSize times
   bool normalizeAffine(const Mat &img, float x, float y, float s, float a11, float a12, float a21, float a22);

   void setAffineShapeCallback(CUAffineShapeCallback *callback)
      {
         affineShapeCallback = callback;
      }

public:
   Mat patch;

protected:
   CUAffineShapeParams par;

private:
   CUAffineShapeCallback *affineShapeCallback;
   std::vector<unsigned char> workspace;
   Mat mask, img, fx, fy;
};

#endif /* AFFINE_H_ */
