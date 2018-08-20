库里我定义了一个名字为：faceDect 的类
构造函数没有任何输入参数，直接 faceDect xx; 即可
只需调用一个方法：
void faceDect::detectMain(cv::Mat& cv_img,vector<int>& faceBox, float& costTime)

cv_img：输入图像。
faceBox：输出人脸框坐标点，只有四个变量，按顺序分别为左上角坐标x1,y1和右下角坐标x2,y2
costTime:算法运行时间。

g++ -I"./include/" -O3 -Wall -c -fmessage-length=0 -std=c++11 -MMD -MP -MF"mtcnn.d" -MT"mtcnn.o" -o "mtcnn.o" "./mtcnn.cpp"
g++ -L"./" -fopenmp -o "ncnn_mtcnn" ./mtcnn.o -lopencv_core -lncnn_mtcnn_so -lopencv_highgui -lopencv_imgproc