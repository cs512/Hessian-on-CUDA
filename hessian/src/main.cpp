/*
 * main.cpp
 *
 *  Created on: 2014-12-3
 *      Author: wangjz
 */

#include <iostream>
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <ctime>

#include "hesaff/pyramid.h"
#include "hesaff/helpers.h"
#include "hesaff/affine.h"
#include "hesaff/siftdesc.h"

#include "pyramid.h"
using namespace std;

#include <cv.h>
#include <opencv2/gpu/gpu.hpp>
using namespace cv;

int main(int argc, char **argv)
{
    PyramidParams par;
    CUHessianDetector cuDet;
    HessianDetector cpDet(par);
    cout << "test of HessianDetector::hessianResponse" << endl;
    {
        const int width = 13000;
//        srand( (unsigned)time( NULL ) );
        Mat testInput(width, width, CV_32FC1, Scalar(0));
        //float *data = testInput.ptr<float>(0);
        srand( (unsigned)time( NULL ) );
        for (int eachRow = 0; eachRow < width; eachRow++)
        {
            for(int eachCol = 0; eachCol < width; eachCol++)
            {
                testInput.at<float>(eachRow, eachCol) = float(rand() % 256);
            }
        }

        float curSigma = par.initialSigma;
        clock_t cuStart = clock();
        Mat cuCur = cuDet.hessianResponse(testInput, curSigma*curSigma);
        clock_t cuEnd = clock();
        Mat cpCur = cpDet.hessianResponse(testInput, curSigma*curSigma);
        clock_t cpEnd = clock();
        int count = 0;
        for (int eachRow = 0; eachRow < width; eachRow++)
        {
            for(int eachCol = 0; eachCol < width; eachCol++)
            {
                //cout << cpCur.at<float>(eachRow, eachCol)<<endl;

                if(cuCur.at<float>(eachRow, eachCol) != cpCur.at<float>(eachRow, eachCol))
                {
                    cout<<"CUDA::CPU"<<endl;
                    cout<<cuCur.at<float>(eachRow, eachCol)<<"::"<<cpCur.at<float>(eachRow, eachCol);
                    cout<<"\t@("<<eachRow<<","<<eachCol<<")"<<endl;
                    ++count;
                }
            }
        }
        if(count == 0)
        {
            cout<<"test pass."<<endl;
            cout<<"CUDA:\t"<<double(cuEnd - cuStart)/CLOCKS_PER_SEC<<"s"<<endl;
            cout<<"CPU:\t"<<double(cpEnd - cuEnd)/CLOCKS_PER_SEC<<"ms"<<endl;
        }
    }
	return 0;
}


