#include "output.h"

Output::Output(cv::Mat image){
    //cvtColor(image, baseImage, CV_GRAY2BGR); // Uncomment this line if the image is in black and white.
    baseImage = image; // Comment this line if the image is in black and white.
}

/**
 * @brief Output::addData Adds a new detected number on the frame.
 * @param x The x position of the number.
 * @param y The y position of the number.
 * @param num The value of the number.
 */
void Output::addData(unsigned int x, unsigned int y, unsigned int num){
    listX.append(x);
    listY.append(y);
    listNum.append(num);
}

/**
 * @brief Output::getImage Returns the image with the numbers on it.
 * @return the image with the detected numbers on it.
 */
cv::Mat Output::getImage(){

    for (int i = 0; i < listX.size(); i++){
        cv::Point p1;
        cv::Point p2;
        cv::Point p3;

        p1.x = listX[i];
        p1.y = listY[i];

        p2.x = listX[i] + 36;
        p2.y = listY[i] + 45;

        p3.x = listX[i];
        p3.y = listY[i] + 45;

        rectangle(baseImage, p1, p2, cv::Scalar(0,0,255));

        putText(baseImage, QString::number(listNum[i]).toStdString(), p3, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0,0,255));
    }

    return baseImage;
}

/**
 * @brief Output::toString Returns a string-representation of the detected numbers.
 * @return a string-representation of the detected numbers, in the same format as the evaluator.
 */
QString Output::toString(){
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
