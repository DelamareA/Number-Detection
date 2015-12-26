#ifndef OUTPUT_H
#define OUTPUT_H

#include <QList>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define MERGE_DISTANCE_X 50
#define MERGE_DISTANCE_Y 20

class Output{

    public:
        Output(cv::Mat);
        cv::Mat getImage();
        void addData(unsigned int x, unsigned int y, unsigned int num);
        QString toString();

    private:
        QList<unsigned int> listX;
        QList<unsigned int> listY;
        QList<unsigned int> listNum;
        cv::Mat baseImage;

};

#endif // OUTPUT_H
