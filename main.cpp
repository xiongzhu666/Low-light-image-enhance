#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
using namespace std;
using namespace cv;
cv::Mat GetKernel(float spatial_sigma, int size)
{
    cv::Mat kernel = Mat_<float>(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float dist = pow(((float)i - float(size/2)),2)+pow(((float)j - float(size/2)),2);
            kernel.at<float>(i,j) = exp(-0.5f * dist  / (spatial_sigma * spatial_sigma));
        }
    }
    return kernel;
}
cv::Mat compute_smoothness_weights(const cv::Mat &L, int x, const cv::Mat &kernel, float eps = 0.01f)
{
    Mat Lp;
    Sobel(L, Lp, CV_32F, int(x==1), int(x==0),1);
    //imwrite("../Lp.png", 255*Lp);
    cv::Mat tmp = Mat::ones(cv::Size (L.cols,L.rows), CV_32F);
    cv::Mat T;
    filter2D(tmp, T, tmp.depth(), kernel,Point(-1,-1),0,BORDER_CONSTANT);
    //imwrite("../T.png", T);
    cv::Mat tt;
    filter2D(Lp, tt, tmp.depth(), kernel,Point(-1,-1),0,BORDER_CONSTANT);
    //imwrite("../tt.png", 255*tt);
    for (int i = 0; i < T.rows; ++i) {
        for (int j = 0; j < T.cols; ++j) {
            T.at<float>(i, j) = T.at<float>(i, j) / (abs(tt.at<float>(i, j))+eps);
            T.at<float>(i, j) = T.at<float>(i, j) / (abs(Lp.at<float>(i, j))+eps);
        }
    }
    return T;
}
std::vector<std::vector<int>> get_sparse_neighbor(int p, int n, int m)
{
    int i = int((float)p/(float)m);
    int j = p%m;
    std::vector<int> d1,d2,d3,d4;
    std::vector<std::vector<int>> d;
    if(i-1>= 0)
    {
        int tmp1 = (i - 1) * m + j;
        d1.push_back(tmp1);
        d1.push_back(i - 1);
        d1.push_back(j);
        d1.push_back(0);
    }
    if(i + 1 < n)
    {
        int tmp = (i + 1) * m + j;
        d2.push_back(tmp);
        d2.push_back(i + 1);
        d2.push_back(j);
        d2.push_back(0);
    }
    if(j - 1 >= 0)
    {
        int tmp = i * m + j - 1;
        d3.push_back(tmp);
        d3.push_back(i);
        d3.push_back(j-1);
        d3.push_back(1);
    }
    if(j + 1 < m)
    {
        int tmp = i * m + j + 1;
        d4.push_back(tmp);
        d4.push_back(i);
        d4.push_back(j+1);
        d4.push_back(1);
    }
    if(d1.size()>0)
    {
        d.push_back(d1);
    }
    if(d2.size()>0)
    {
        d.push_back(d2);
    }
    if(d3.size()>0)
    {
        d.push_back(d3);
    }
    if(d4.size()>0)
    {
        d.push_back(d4);
    }
    return d;
}
int main() {
    std::cout << "Hello, World!" << std::endl;
    Mat im = imread("../2.jpg",CV_32F);
    im.convertTo(im, CV_32FC3, 1.0/255, 0);
    int h = im.rows;
    int w = im.cols;
    cv::Mat L = Mat::zeros(cv::Size (w,h), CV_32F);
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            float max;
            max = im.at<cv::Vec3f>(i,j)[0] > im.at<cv::Vec3f>(i,j)[1] ? im.at<cv::Vec3f>(i,j)[0] : im.at<cv::Vec3f>(i,j)[1];
            max = max > im.at<cv::Vec3f>(i,j)[2] ? max : im.at<cv::Vec3f>(i,j)[2];
            L.at<float>(i, j) = max;
        }
    }
    cv::Mat ken = GetKernel(3.f,15);
//    imwrite("../L.png", 255*L);
    Mat wx = compute_smoothness_weights(L,1,ken);
    //imwrite("../wx.png", wx);
    Mat wy = compute_smoothness_weights(L,0,ken);
    //imwrite("../wy.png", wy);
    std::vector<float> L_1d;
    for (int l = 0; l < L.rows; ++l) {
        for (int i = 0; i < L.cols; ++i) {
            L_1d.push_back(L.at<float>(l, i));
        }
    }
    vector<int> row,column;
    vector<float> data;
    std::cout << h*w << endl;
    for (int m = 0; m < h*w; ++m) {
        double diag = 0.f;
        std::vector<std::vector<int>> vec = get_sparse_neighbor(m,w,h);
        for (int i = 0; i < vec.size(); ++i) {
            float weight;
            if(vec[i][3] != 0)
            {
                weight = wx.at<float>(vec[i][1], vec[i][2]);
            } else
            {
                weight = wy.at<float>(vec[i][1], vec[i][2]);
            }
            row.push_back(m);
            column.push_back(vec[i][0]);
            data.push_back(-weight);
            diag += weight;
        }
        //std::cout  << setprecision(20) << diag << endl;
        row.push_back(m);
        column.push_back(m);
        data.push_back(diag);
    }
    std::cout << data.size() << endl;
//    for (int i = 0; i < data.size(); ++i) {
//        cout << data[i] << endl;
//    }
    std::vector<std::vector<int>> vec = get_sparse_neighbor(1,500,500);
    cout << vec[0][0]<< " " << vec[0][1] << " " << vec[0][2] << " " <<  vec[0][3] << endl;
    cout << vec[1][0]<< " " << vec[1][1] << " " << vec[1][2] << " " << vec[1][3] << endl;
    cout << vec[2][0]<< " " << vec[2][1] << " " << vec[2][2] << " " << vec[2][3] << endl;
    return 0;
}
