/*
 * ncnn_mtcnn_so.hpp
 *
 *  Created on: Aug 10, 2018
 *      Author: hans
 */

#ifndef NCNN_MTCNN_SO_HPP_
#define NCNN_MTCNN_SO_HPP_
#include <vector>
#include <opencv2/opencv.hpp>
#include "net.h"
using namespace std;
using namespace cv;
struct Bbox{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    float ppoint[10];
    float regreCoord[4];
};
struct orderScore{
    float score;
    int oriOrder;
};
class faceDect{
public:
	faceDect();
    ~faceDect();
    int detectMain(cv::Mat& cv_img, vector<int>& faceBox, float& costTime);
private:
    float calIOU(Bbox &box1,Bbox &box2,const string& modelname);
    void detect(ncnn::Mat& img_, vector<Bbox>& finalBbox);
    void generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, vector<orderScore>& bboxScore_, float scale);
    void nms(vector<Bbox> &boundingBox_, vector<orderScore> &bboxScore_, const float overlap_threshold, const string& modelname);
    void refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width);
    float rou(float src, int bits);
    ncnn::Net Pnet, Rnet, Onet;
    ncnn::Mat img;
    const float nms_threshold[3] = {0.5, 0.7, 0.7};
    const float threshold[3] = {0.8, 0.8, 0.6};
    const float mean_vals[3] = {127.5, 127.5, 127.5};
    const float norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};
    vector<Bbox> firstBbox_, secondBbox_,thirdBbox_;
    vector<orderScore> firstOrderScore_, secondBboxScore_, thirdBboxScore_;
    int MIN_DET_SIZE = 12;
	int minsize = 90;
	int threads = 4;
	Bbox LaFaceBox,CuFaceBox,rectangle,MeanNose,MeanFaceBox;
	vector< pair<int,int>> NoseList,CuFaceBoxList;
	vector<Bbox> rectangles;
	float IOUthres = 0.95;
	int MeanFrame = 5;
	int pad=0;
	string MinIOU="Min";
	string UnionIOU="Union";
};
#endif /* NCNN_MTCNN_SO_HPP_ */
