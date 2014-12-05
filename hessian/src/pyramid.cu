/*
 * pyramid.cu
 *
 *  Created on: 2014-12-4
 *      Author: wangjz
 */

#include <cv.h>
#include "pyramid.h"

using namespace cv;

__global__ void performHessianResponse(float *in, float *out, float norm, int cols, int rows)
{
    float v11, v12, v13, v21, v22, v23, v31, v32, v33;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if(x > cols - 3)
        return;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(y > rows - 3)
            return;
    int offset = x + y * cols;
    /* fill in shift registers at the beginning of the row */
    /* fetch remaining values (last column) */
    v11 = in[offset           ]; v12 = in[offset            + 1]; v13 = in[offset            + 2];
    v21 = in[offset + cols    ]; v22 = in[offset + cols     + 1]; v23 = in[offset + cols     + 2];
    v31 = in[offset + 2 * cols]; v32 = in[offset + 2 * cols + 1]; v33 = in[offset + 2 * cols + 2];
    // compute 3x3 Hessian values from symmetric differences.
    float Lxx = (v21 - 2*v22 + v23);
    float Lyy = (v12 - 2*v22 + v32);
    float Lxy = (v13 - v11 + v31 - v33)/4.0f;

    /* normalize and write out */
    out[offset + cols + 1] = (Lxx * Lyy - Lxy * Lxy)*norm;
}

Mat CUHessianDetector::hessianResponse(const Mat &inputImage, float norm)
{
   const int rows = inputImage.rows;
   const int cols = inputImage.cols;

   // allocate output
   //Mat outputImage(rows, cols, CV_32FC1);
   float * gpuOutputImage = NULL;
   float * gpuInputImage = NULL;
   // TODO handle error
   cudaMalloc((void**)&gpuOutputImage, (size_t)(rows*cols*sizeof(float)));
   cudaMalloc((void**)&gpuInputImage, (size_t)(rows*cols*sizeof(float)));
   cudaMemset((void*)gpuOutputImage, 0, rows*cols*sizeof(float));
   cudaMemcpy((void*)gpuInputImage, (void*)inputImage.ptr<float>(0), rows*cols*sizeof(float), cudaMemcpyHostToDevice);
   float norm2 = norm * norm;
   dim3 blocks((15 - 2 + cols)/16, (15 - 2 + rows)/16);
   dim3 threads(16, 16);
   Mat outputImage(rows, cols, CV_32FC1);
   performHessianResponse<<<blocks, threads>>>(gpuInputImage, gpuOutputImage, norm2, cols, rows);
   cudaMemcpy((void*)outputImage.ptr<float>(0), (void*)gpuOutputImage, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
   cudaFree(gpuOutputImage);
   cudaFree(gpuInputImage);
   return outputImage;
}

