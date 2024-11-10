#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

//均值滤波，均匀图像
cv::Mat meanFilter(int size) {
    cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) / (float)(size * size);  
    return kernel;
}
//sobel算子, axis为x或y,检测图像边缘
cv::Mat sobelFilter(char axis){
    cv::Mat kernel;
    if (axis == 'x') {
        kernel = (cv::Mat_<float>(3, 3) << -1, 0, 1,
                                           -2, 0, 2,
                                           -1, 0, 1);
    } else if (axis == 'y') {
        kernel = (cv::Mat_<float>(3, 3) << -1, -2, -1,
                                            0,  0,  0,
                                            1,  2,  1);
    }
    return kernel;
}

cv::Mat convolution2D(cv::Mat& image, cv::Mat& kernel){

    int image_height = image.rows;
    int image_width = image.cols;
    int kernel_height = kernel.rows;
    int kernel_width = kernel.cols;

    cv::Mat output_image = cv::Mat::zeros(image_height - kernel_height + 1, image_width - kernel_width + 1, CV_32F);

    for(int i = 0; i < output_image.rows; i++){
        for(int j = 0; j < output_image.cols; j++){
            for(int m = 0; m < kernel_height; m++){
                for(int n = 0; n < kernel_width; n++){
                    output_image.at<int>(i, j) += image.at<int>(i + m, j + n) * kernel.at<int>(m, n);
                }
            }
        }
    }

    return output_image;
}

int main(){
    cv::Mat image = cv::imread("/home/yst/文档/jwj/python/my/test.jpg");
    int size = 3;
    cv::Mat mean_kernel = meanFilter(size);
    cv::Mat mean_image = convolution2D(image, mean_kernel);

    cv::Mat sobel_x = sobelFilter('x');
    cv::Mat sobel_image = convolution2D(image, sobel_x);

    cv::imwrite("/home/yst/文档/jwj/python/my/dst.jpg", mean_image);
    return 0;
}