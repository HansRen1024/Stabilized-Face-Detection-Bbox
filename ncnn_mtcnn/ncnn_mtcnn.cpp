#include <vector>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iomanip>
#include "net.h"
#include "models/det1.mem.h"
#include "models/det2.mem.h"
#include "models/det3.mem.h"
#include "models/det1.id.h"
#include "models/det2.id.h"
#include "models/det3.id.h"
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
bool sortScore(Bbox& lsh, Bbox& rsh){
	return lsh.score>rsh.score;
}
bool cmpScore(orderScore lsh, orderScore rsh){
    if(lsh.score<rsh.score)return true;
    else return false;
}
static float getElapse(struct timeval *tv1,struct timeval *tv2){
    float t = 0.0f;
    if (tv1->tv_sec == tv2->tv_sec)t = (tv2->tv_usec - tv1->tv_usec)/1000.0f;
    else t = ((tv2->tv_sec - tv1->tv_sec) * 1000 * 1000 + tv2->tv_usec - tv1->tv_usec)/1000.0f;
    return t;
}
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
	string MinIOU="Min";
	string UnionIOU="Union";
	Bbox LaFaceBox,CuFaceBox,rectangle,MeanNose,MeanFaceBox;
	vector< pair<int,int>> NoseList,CuFaceBoxList;
	std::vector<int> padList;
	vector<Bbox> rectangles;
	float IOUthres = 0.95;
	int MeanFrame = 5;
	int pad=0;
};
faceDect::faceDect(){
    Pnet.load_param(det1_param_bin);
    Pnet.load_model(det1_bin);
    Rnet.load_param(det2_param_bin);
    Rnet.load_model(det2_bin);
    Onet.load_param(det3_param_bin);
    Onet.load_model(det3_bin);
//    Pnet.load_param("./models/det1.param");
//    Pnet.load_model("./models/det1.bin");
//    Rnet.load_param("./models/det2.param");
//    Rnet.load_model("./models/det2.bin");
//    Onet.load_param("./models/det3.param");
//    Onet.load_model("./models/det3.bin");
}
faceDect::~faceDect(){
	Pnet.clear();
	Rnet.clear();
	Onet.clear();
}
float faceDect::calIOU(Bbox &box1,Bbox &box2,const string& modelname){
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	float startx = 0;
	float endx = 0;
	float starty = 0;
	float endy = 0;
	float width = 0;
	float height = 0;
	endx = (box1.x2>box2.x2)?box1.x2:box2.x2;
	startx = (box1.x1>box2.x1)?box1.x1:box2.x1;
	endy = (box1.y2>box2.y2)?box1.y2:box2.y2;
	starty = (box1.y1>box2.y1)?box1.y1:box2.y1;
	width = (box1.x2-box1.x1)+(box2.x2-box2.x1)-(endx-startx);
	height = (box1.y2-box1.y1)+(box2.y2-box2.y1)-(endy-starty);
	if (width>0 and height>0){
		maxX = startx;
		maxY = starty;
		minX = (box1.x2<box2.x2)?box1.x2:box2.x2;
		minY = (box1.y2<box2.y2)?box1.y2:box2.y2;
		if(!modelname.compare("Union")){
			maxX = (minX+1>maxX)?(minX-maxX+1):(maxX-minX+1);
			maxY = (minY+1>maxY)?(minY-maxY+1):(maxY-minY+1);
		}
		else{
			maxX = ((minX-maxX+1)>0)?(minX-maxX+1):0;
			maxY = ((minY-maxY+1)>0)?(minY-maxY+1):0;
		}
		IOU = maxX * maxY;
		if(!modelname.compare("Union"))IOU = IOU/(box1.area + box2.area - IOU);
		else if(!modelname.compare("Min"))IOU = IOU/((box1.area<box2.area)?box1.area:box2.area);
	}
	return IOU;
}
void faceDect::generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, vector<orderScore>& bboxScore_, float scale){
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    float *p = score.channel(1);//score.data + score.cstep;
    float *plocal = location.data;
    Bbox bbox;
    orderScore order;
    for(int row=0;row<score.h;row++){
        for(int col=0;col<score.w;col++){
            if(*p>threshold[0]){
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride*col+1)/scale);
                bbox.y1 = round((stride*row+1)/scale);
                bbox.x2 = round((stride*col+1+cellsize)/scale);
                bbox.y2 = round((stride*row+1+cellsize)/scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=location.channel(channel)[0];
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal++;
        }
    }
}
void faceDect::nms(vector<Bbox> &boundingBox_, vector<orderScore> &bboxScore_, const float overlap_threshold, const string& modelname){
    if(boundingBox_.empty())return;
    vector<int> heros;
    //sort the score
    sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);
    int order = 0;
    float IOU = 0;
    while(bboxScore_.size()>0){
        order = bboxScore_.back().oriOrder;
        bboxScore_.pop_back();
        if(order<0)continue;
        if(boundingBox_.at(order).exist == false) continue;
        heros.push_back(order);
        boundingBox_.at(order).exist = false;//delete it
        for(int num=0;num<int(boundingBox_.size());num++){
            if(boundingBox_.at(num).exist){
            	IOU = calIOU(boundingBox_.at(num),boundingBox_.at(order),modelname);
                if(IOU>overlap_threshold){
                    boundingBox_.at(num).exist=false;
                    for(vector<orderScore>::iterator it=bboxScore_.begin(); it!=bboxScore_.end();it++){
                        if((*it).oriOrder == num) {
                            (*it).oriOrder = -1;
                            break;
                        }
                    }
                }
            }
        }
    }
    for(uint i=0;i<heros.size();i++)
        boundingBox_.at(heros.at(i)).exist = true;
}
void faceDect::refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width){
    if(vecBbox.empty())return;
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++){
        if((*it).exist){
            bbw = (*it).x2 - (*it).x1 + 1;
            bbh = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[0]*bbw;
            y1 = (*it).y1 + (*it).regreCoord[1]*bbh;
            x2 = (*it).x2 + (*it).regreCoord[2]*bbw;
            y2 = (*it).y2 + (*it).regreCoord[3]*bbh;
            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            maxSide = (h>w)?h:w;
            x1 = x1 + w*0.5 - maxSide*0.5;
            y1 = y1 + h*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);
            //boundary check
            if((*it).x1<0)(*it).x1=0;
            if((*it).y1<0)(*it).y1=0;
            if((*it).x2>width)(*it).x2 = width - 1;
            if((*it).y2>height)(*it).y2 = height - 1;
            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
        }
    }
}
void faceDect::detect(ncnn::Mat& img_, vector<Bbox>& finalBbox_){
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();
    img = img_;
    int img_w, img_h;
    img_w = img.w;
    img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);
    float minl = img_w<img_h?img_w:img_h;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    int factor_count = 0;
    float factor = 0.709;
    vector<float> scales_;
    while(minl>MIN_DET_SIZE){
        if(factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    orderScore order;
    int count = 0;
    for (size_t i = 0; i < scales_.size(); i++) {
        int hs = (int)ceil(img_h*scales_[i]);
        int ws = (int)ceil(img_w*scales_[i]);
        //ncnn::Mat in = ncnn::Mat::from_pixels_resize(image_data, ncnn::Mat::PIXEL_RGB2BGR, img_w, img_h, ws, hs);
        ncnn::Mat in;
        resize_bilinear(img_, in, ws, hs);
        //in.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = Pnet.create_extractor();
        ex.set_light_mode(true);
        ex.set_num_threads(threads);
        ex.input(det1_param_id::BLOB_data, in);
		ncnn::Mat score_, location_;
		ex.extract(det1_param_id::BLOB_prob1, score_);
		ex.extract(det1_param_id::BLOB_conv4_2, location_);
//		printf( "%s = %g \n ", "score_", score_[0]);
//        ex.input("data", in);
//        ncnn::Mat score_, location_;
//        ex.extract("prob1", score_);
//        ex.extract("conv4-2", location_);
        vector<Bbox> boundingBox_;
        vector<orderScore> bboxScore_;
        generateBbox(score_, location_, boundingBox_, bboxScore_, scales_[i]);
        nms(boundingBox_, bboxScore_, nms_threshold[0], MinIOU);
        for(vector<Bbox>::iterator it=boundingBox_.begin(); it!=boundingBox_.end();it++){
            if((*it).exist){
                firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
        bboxScore_.clear();
        boundingBox_.clear();
    }
    if(count<1)return;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0], MinIOU);
    refineAndSquareBbox(firstBbox_, img_h, img_w);
    //second stage
    count = 0;
    for(vector<Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            ncnn::Mat tempIm;
            copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
            ncnn::Mat in;
            resize_bilinear(tempIm, in, 24, 24);
            ncnn::Extractor ex = Rnet.create_extractor();
            ex.set_light_mode(true);
            ex.set_num_threads(threads);
            ex.input(det2_param_id::BLOB_data, in);
			ncnn::Mat score, bbox;
			ex.extract(det2_param_id::BLOB_prob1, score);
			ex.extract(det2_param_id::BLOB_conv5_2, bbox);
//            ex.input("data", in);
//            ncnn::Mat score, bbox;
//            ex.extract("prob1", score);
//            ex.extract("conv5-2", bbox);
            if(*(score.data+score.cstep)>threshold[1]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=bbox.channel(channel)[0];//*(bbox.data+channel*bbox.cstep);
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score.channel(1)[0];//*(score.data+score.cstep);
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else (*it).exist=false;
        }
    }
    if(count<1)return;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1], MinIOU);
    refineAndSquareBbox(secondBbox_, img_h, img_w);
    //third stage
    count = 0;
    for(vector<Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            ncnn::Mat tempIm;
            copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
            ncnn::Mat in;
            resize_bilinear(tempIm, in, 48, 48);
            ncnn::Extractor ex = Onet.create_extractor();
            ex.set_light_mode(true);
            ex.set_num_threads(threads);
            ex.input(det3_param_id::BLOB_data, in);
			ncnn::Mat score, bbox, keyPoint;
			ex.extract(det3_param_id::BLOB_prob1, score);
			ex.extract(det3_param_id::BLOB_conv6_2, bbox);
			ex.extract(det3_param_id::BLOB_conv6_3, keyPoint);
//            ex.input("data", in);
//            ncnn::Mat score, bbox, keyPoint;
//            ex.extract("prob1", score);
//            ex.extract("conv6-2", bbox);
//            ex.extract("conv6-3", keyPoint);
            if(score.channel(1)[0]>threshold[2]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=bbox.channel(channel)[0];
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score.channel(1)[0];
                for(int num=0;num<5;num++){
                    (it->ppoint)[num] = it->x1 + (it->x2 - it->x1)*keyPoint.channel(num)[0];
                    (it->ppoint)[num+5] = it->y1 + (it->y2 - it->y1)*keyPoint.channel(num+5)[0];
                }
                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else (*it).exist=false;
		}
	}
    if(count<1)return;
    refineAndSquareBbox(thirdBbox_, img_h, img_w);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], MinIOU);
    finalBbox_ = thirdBbox_;
}
float faceDect::rou(float src, int bits){
	stringstream ss;
	ss << fixed << setprecision(bits) << src;
	ss >> src;
	return src;
}
int faceDect::detectMain(cv::Mat& cv_img, vector<int>& faceBox, float& costTime){
	faceBox.clear();
	rectangles.clear();
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR2RGB, cv_img.cols, cv_img.rows);
	struct timeval  tv1,tv2;
	struct timezone tz1,tz2;
	gettimeofday(&tv1,&tz1);
	detect(ncnn_img, rectangles);
	if (int(rectangles.size())==0){
		LaFaceBox.exist=false;
		return 0;
	}
	sort(rectangles.begin(),rectangles.end(),sortScore);
	rectangle = rectangles[0];
	float rouRate  = rou((rectangle.x2-rectangle.x1)/float(cv_img.cols),1);
	pad = round(0.7*rouRate*cv_img.cols/2);
	padList.push_back(pad);
	NoseList.push_back(make_pair(round((rectangle.x2+rectangle.x1)/2),round((rectangle.y2+rectangle.y1)/2)));
	while (int(NoseList.size())>MeanFrame){
		NoseList.erase(NoseList.begin());
		padList.erase(padList.begin());
	}
	int MeanPad=0;
	for(vector<int>::iterator it=padList.begin(); it!=padList.end();it++){
		MeanPad+=int(round((*it)/float(padList.size())));
	}
	padList.push_back(MeanPad);
	MeanNose.x1=0;
	MeanNose.y1=0;
	for(vector< pair<int,int>>::iterator it=NoseList.begin(); it!=NoseList.end();it++){
		MeanNose.x1+=int(round((*it).first/float(NoseList.size())));
		MeanNose.y1+=int(round((*it).second/float(NoseList.size())));
	}
	NoseList.push_back(make_pair(MeanNose.x1,MeanNose.y1));
	CuFaceBox.x1 = MeanNose.x1-MeanPad;
	CuFaceBox.y1 = MeanNose.y1-MeanPad;
	CuFaceBox.x2 = MeanNose.x1+MeanPad;
	CuFaceBox.y2 = MeanNose.y1+MeanPad;
	if(!LaFaceBox.exist){
		LaFaceBox=CuFaceBox;
		LaFaceBox.exist=true;
	}
	LaFaceBox.area = (LaFaceBox.x2-LaFaceBox.x1)*(LaFaceBox.y2-LaFaceBox.y1);
	CuFaceBox.area = (CuFaceBox.x2-CuFaceBox.x1)*(CuFaceBox.y2-CuFaceBox.y1);
	float IOUrate = calIOU(LaFaceBox,CuFaceBox,UnionIOU);
	if (IOUrate>IOUthres)CuFaceBox=LaFaceBox;
	LaFaceBox=CuFaceBox;
	CuFaceBoxList.push_back(make_pair(CuFaceBox.x1,CuFaceBox.y1));
	while (int(CuFaceBoxList.size())>MeanFrame)CuFaceBoxList.erase(CuFaceBoxList.begin());
	MeanFaceBox.x1=0;
	MeanFaceBox.y1=0;
	for(vector< pair<int,int>>::iterator it=CuFaceBoxList.begin(); it!=CuFaceBoxList.end();it++){
		MeanFaceBox.x1+=int(round((*it).first/float(CuFaceBoxList.size())));
		MeanFaceBox.y1+=int(round((*it).second/float(CuFaceBoxList.size())));
	}
	CuFaceBoxList.push_back(make_pair(MeanFaceBox.x1,MeanFaceBox.y1));
	faceBox.resize(4);
	faceBox[0] = MeanFaceBox.x1;
	faceBox[1] = MeanFaceBox.y1;
	faceBox[2] = MeanFaceBox.x1+MeanPad*2;
	faceBox[3] = MeanFaceBox.y1+MeanPad*2;
	gettimeofday(&tv2,&tz2);
	costTime = getElapse(&tv1, &tv2);
	return 0;
}
void cam(){
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
//		printf( "%s = %g ms \n ", "Detection All time", costTime);
	}
	capture.release();
}
void picture(){
	std::string imgPath = "face.jpg";
	faceDect xx;
	std::vector<int> finalBbox;
	float costTime;
	cv::Mat frame = cv::imread(imgPath);
	xx.detectMain(frame,finalBbox,costTime);
	if(finalBbox.size()>0){
		cv::rectangle(frame, Point(finalBbox[0], finalBbox[1]), Point(finalBbox[2], finalBbox[3]), Scalar(0,0,255), 2,8,0);
	}
	printf("Cost Time: %g\n",costTime);
	cv::imshow("SSD", frame);
	cv::waitKey(0);
}
int main(int argc, char** argv){
	cam();
//	picture();
	cv::destroyAllWindows();
	return 0;
}
