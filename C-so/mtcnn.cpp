#include <vector>
#include <opencv2/opencv.hpp>
#include "./include/ncnn_mtcnn_so.hpp"
using namespace std;
using namespace cv;
int main(int argc, char** argv){
	cv::VideoCapture capture(0);
	cv::Mat cv_img;
	faceDect xx;
	std::vector<int> finalBbox;
	float costTime;
	while(true){
		capture >> cv_img;
		cv::resize(cv_img,cv_img,cv::Size(640,480));
		xx.detectMain(cv_img,finalBbox,costTime);
		if(finalBbox.size()>0)cv::rectangle(cv_img, Point(finalBbox[0], finalBbox[1]), Point(finalBbox[2], finalBbox[3]), Scalar(0,0,255), 2,8,0);
		cv::imshow("SSD", cv_img);
		int key = cv::waitKey(1);
		if (key == 27)break;
		printf( "%s = %g ms \n ", "Detection All time", costTime);
		// printf( "x1: %d, y1: %d, x2: %d, y2: %d\n", finalBbox[0], finalBbox[1], finalBbox[2], finalBbox[3]);
	}
	return 0;
}
