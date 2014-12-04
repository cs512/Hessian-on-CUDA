/*
 * pyramid.cu
 *
 *  Created on: 2014-12-4
 *      Author: wangjz
 */

#include <cv.h>
#include "pyramid.h"

using namespace cv;

__global__ void performHessianResponse(const float *in, float *out, float norm)
{

}

Mat CUHessianDetector::hessianResponse(const Mat &inputImage, float norm)
{
   const int rows = inputImage.rows;
   const int cols = inputImage.cols;
   const int stride = cols;

   // allocate output
   //Mat outputImage(rows, cols, CV_32FC1);
   float * gpuOutputImage = NULL;
   float * gpuInputImage = NULL;
   // TODO handle error
   cudaMalloc<float>(gpuOutputImage, rows*cols);
   cudaMalloc<float>(gpuInputImage, rows*cols);
   cudaMemcpy((void*)gpuInputImage, (void*)inputImage.ptr<float>(0), rows*cols*sizeof(float), cudaMemcpyHostToDevice);
   // setup input and output pointer to be centered at 1,0 and 1,1 resp.
   // need pointer conv
   //const float *in = inputImage.ptr<float>(1);

   //float *out = outputImage.ptr<float>(1) + 1;
   //float *gpuOut = gpuOutputImage[cols + 1];

   float norm2 = norm * norm;
   // TODO CUDAble
   /* move 3x3 window and convolve */
   for (int r = 1; r < rows - 1; ++r)
   {
      float v11, v12, v21, v22, v31, v32;
      /* fill in shift registers at the beginning of the row */
      v11 = in[-stride]; v12 = in[1 - stride];
      v21 = in[      0]; v22 = in[1         ];
      v31 = in[+stride]; v32 = in[1 + stride];
      /* move input pointer to (1,2) of the 3x3 square */
      in += 2;
      for (int c = 1; c < cols - 1; ++c)
      {
         /* fetch remaining values (last column) */
         const float v13 = in[-stride];
         const float v23 = *in;
         const float v33 = in[+stride];

         // compute 3x3 Hessian values from symmetric differences.
         float Lxx = (v21 - 2*v22 + v23);
         float Lyy = (v12 - 2*v22 + v32);
         float Lxy = (v13 - v11 + v31 - v33)/4.0f;

         /* normalize and write out */
         *out = (Lxx * Lyy - Lxy * Lxy)*norm2;

         /* move window */
         v11=v12; v12=v13;
         v21=v22; v22=v23;
         v31=v32; v32=v33;

         /* move input/output pointers */
         in++; out++;
      }
      out += 2;
   }
   return outputImage;
}

