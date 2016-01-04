#include "frameoutput.h"
#include <QDebug>

/**
 * @brief FrameOutput::FrameOutput Main constructor for the output of a frame.
 * @param image The image of the frame.
 */
FrameOutput::FrameOutput(cv::Mat image){
    //cvtColor(image, baseImage, CV_GRAY2BGR); // Uncomment this line if the image is in black and white.
    baseImage = image.clone(); // Comment this line if the image is in black and white.
}

/**
 * @brief Output::addData Adds a new detected number on the frame.
 * @param x The x position of the number.
 * @param y The y position of the number.
 * @param num The value of the number.
 */
void FrameOutput::addData(unsigned int x, unsigned int y, unsigned int num){
    listX.append(x);
    listY.append(y);
    listNum.append(num);
}

/**
 * @brief FrameOutput::removeData Removes all existing data, if any, at the given position.
 * @param x The x position of the number.
 * @param y The y position of the number.
 * @param maxDistance The maximum distance of the data and x and y for the data to be removed.
 */
void FrameOutput::removeData(unsigned int x, unsigned int y, unsigned int maxDistance){
    for (int i = 0; i < listX.size(); i++){
        if ((listX[i] - x)*(listX[i] - x) + (listY[i] - y)*(listY[i] - y) < maxDistance*maxDistance){
            listX.removeAt(i);
            listY.removeAt(i);
            listNum.removeAt(i);

            i--;
        }
    }
}

/**
 * @brief FrameOutput::isDataClose Indicates if a number is close to the input position.
 * @param x The x position of the number.
 * @param y The y position of the number.
 * @param maxDistance
 * @return The maximum distance of the data and x and y for return value to be true.
 */
bool FrameOutput::isDataClose(unsigned int x, unsigned int y, unsigned int maxDistance){
    for (int i = 0; i < listX.size(); i++){
        if ((listX[i] - x)*(listX[i] - x) + (listY[i] - y)*(listY[i] - y) < maxDistance*maxDistance){
            return true;
        }
    }

    return false;
}

/**
 * @brief FrameOutput::getAllData Get all of the current data.
 * @return A list containing the positio of the datas in the frame.
 */
QList<cv::Point2i> FrameOutput::getAllData(){
    QList<cv::Point2i> result;

    for (int i = 0; i < listX.size(); i++){
        cv::Point2i pt;
        pt.x = listX[i];
        pt.y = listY[i];

        result.push_back(pt);
    }

    return result;
}

/**
 * @brief Output::getImage Returns the image with the numbers on it.
 * @return the image with the detected numbers on it.
 */
cv::Mat FrameOutput::getImage(){

    for (int i = 0; i < listX.size(); i++){
        cv::Point p1;
        cv::Point p2;
        cv::Point p3;
        cv::Point p4;
        cv::Point p5;

        p1.x = listX[i];
        p1.y = listY[i];

        p2.x = listX[i] + 44;
        p2.y = listY[i] + 52;

        p3.x = listX[i];
        p3.y = listY[i] + 52;

        p4.x = listX[i] + 22;
        p4.y = listY[i] + 26;

        p5.x = listX[i] + 44;
        p5.y = listY[i] + 26;

        line(baseImage, p1, p3, cv::Scalar(0,0,255), 1);
        line(baseImage, p1, p4, cv::Scalar(0,0,255), 1);
        line(baseImage, p4, p5, cv::Scalar(0,0,255), 1);
        line(baseImage, p5, p2, cv::Scalar(0,0,255), 1);
        line(baseImage, p3, p2, cv::Scalar(0,0,255), 1);

        putText(baseImage, QString::number(listNum[i]).toStdString(), p3, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0,0,255), 2);
    }

    return baseImage;
}

/**
 * @brief Output::toString Returns a string-representation of the detected numbers.
 * @return a string-representation of the detected numbers, in the same format as the evaluator.
 */
QString FrameOutput::toString(){
    QString out;
    out += QString::number(listX.size());
    out += '%';

    for (int i = 0; i < listX.size(); i++){
        out += QString::number(listNum[i]) + "#" + QString::number(listX[i]) + "#" + QString::number(listY[i]);
        out += '%';
    }

    out += '@';

    return out;
}
