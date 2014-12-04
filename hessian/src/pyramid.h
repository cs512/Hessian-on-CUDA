/*
 * pyramid.h
 *
 *  Created on: 2014-12-4
 *      Author: wangjz
 */

#ifndef PYRAMID_H_
#define PYRAMID_H_
#include <cv.h>
using namespace cv;

class CUHessianDetector {
public:
    Mat hessianResponse(const Mat &inputImage, float norm);

private:
};


#endif /* PYRAMID_H_ */
