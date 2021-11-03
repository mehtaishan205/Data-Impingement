#include <stdio.h>
#include <cmath>
#include <math.h>

#include <opencv2/opencv.hpp>
using namespace std;

int main()
{
    int count = 0;

    cv::Mat src_8uc3_img = cv::imread("newwe.png", cv::IMREAD_GRAYSCALE);

    if (src_8uc3_img.empty()) {
        printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
    }

    cv::Mat gray_32fc1_img;
    src_8uc3_img.convertTo(gray_32fc1_img, CV_32FC1, 1.0 / 255.0);

    cv::Mat dst;
    //cv::Mat dst = gray_32fc1_img.clone();
    gray_32fc1_img.copyTo(dst);



    for (int i = 1; i < gray_32fc1_img.rows - 1; i++) {
        for (int j = 1; j < gray_32fc1_img.cols - 1; j++) {

            if (dst.at<float>(i, j) == 0)
            {
                float cN, cS, cE, cW, cne, cnw, cse, csw;
                float deltacN, deltacS, deltacE, deltacW, deltacnw, deltacne, deltacsw, deltacse;
                float div;
                float su, wei;
                deltacN = (dst.at<float>(i, j - 1) - dst.at<float>(i, j));
                deltacS = (dst.at<float>(i, j + 1) - dst.at<float>(i, j));
                deltacE = (dst.at<float>(i + 1, j) - dst.at<float>(i, j));
                deltacW = (dst.at<float>(i - 1, j) - dst.at<float>(i, j));
                deltacnw = (dst.at<float>(i - 1, j - 1) - dst.at<float>(i, j));
                deltacne = (dst.at<float>(i + 1, j - 1) - dst.at<float>(i, j));
                deltacsw = (dst.at<float>(i - 1, j + 1) - dst.at<float>(i, j));
                deltacse = (dst.at<float>(i + 1, j + 1) - dst.at<float>(i, j));
                div = abs(deltacS + deltacN) + abs(deltacE + deltacW) + abs(deltacse + deltacnw) + abs(deltacsw + deltacne);

                if (div > 0.6)
                {
                    count++;
                    su = 0;
                    wei = 0;

                    for (int r = -2; r <= 2; r++)
                    {
                        for (int c = -2; c <= 2; c++)
                        {
                            if (i + r >= 0 && i + r <= gray_32fc1_img.rows - 1 && j + c >= 0 && j + c <= gray_32fc1_img.cols - 1 && dst.at<float>(i + r, j + c) != 0)
                            {
                                wei += 1 / sqrt((r * r) + (c * c));
                                su += (dst.at<float>(i + r, j + c) / sqrt((r * r) + (c * c)));
                            }
                        }
                    }
                    if (su != 0)
                        dst.at<float>(i, j) = su / wei;
                }
            }

        }
    }


    cv::imshow("Original Valve gray", gray_32fc1_img);

    cv::imshow("Filtered Valve gray", dst);
    dst.convertTo(dst, CV_8U,255.0/1.0);
    cv::imwrite("guassianrecovered.png", dst);
    cv::waitKey(0); // wait until keypressed
    std::cout << count;
    return 0;
}