#include "functions.h"
#include <QDebug>

int mostProbableNumber(cv::Mat image, QList<int> digitsOnField){
    cv::Mat blurredImage;
    cv::Mat hsv;
    cv::Mat final;
    GaussianBlur(image, blurredImage, cv::Size(3, 3), 0, 0);
    cvtColor(blurredImage, hsv, CV_BGR2HSV);
    cvtColor(image, final, CV_BGR2GRAY);

    cv::Mat colorSeg;
    pyrMeanShiftFiltering(blurredImage, colorSeg, 10, 18, 3);

    cv::Mat colorSegHSV;
    cvtColor(colorSeg, colorSegHSV, CV_BGR2HSV);

    //cv::imshow("Hello", colorSeg);
    //cv::waitKey(40000 );

    cv::Mat imageLab;
    cvtColor(colorSeg, imageLab, CV_BGR2Lab);

    cv::Scalar mean = cv::mean(imageLab);

    cv::Mat saliency;
    cvtColor(image, saliency, CV_BGR2GRAY);

    for (int x = 0; x < imageLab.cols; x++){
        for (int y = 0; y < imageLab.rows; y++){
            cv::Vec3b intensity = imageLab.at<cv::Vec3b>(y, x);
            uchar value = sqrt(pow((intensity[0] - mean[0]),2) + pow((intensity[1] - mean[1]),2) + pow((intensity[2] - mean[2]),2));
            saliency.at<uchar>(y, x) = value;

            if (value > 30){
                final.at<uchar>(y, x) = 255;
            }
            else {
                final.at<uchar>(y, x) = 0;
            }
        }
    }

    /*int maxV = 0;
    int maxS = 0;
    long meanV = 0;
    long meanS = 0;
    for (int x = 0; x < image.cols; x++){
        for (int y = 0; y < image.rows; y++){
            cv::Vec3b intensity = colorSegHSV.at<cv::Vec3b>(y, x);

            meanV += intensity[2];
            meanS += intensity[1];

            if (intensity[2] > maxV){
                maxV = intensity[2];
            }

            if (intensity[1] > maxS){
                maxS = intensity[1];
            }
        }
    }

    meanV /= (image.cols * image.rows);
    meanS /= (image.cols * image.rows);

    for (int x = 0; x < image.cols; x++){
        for (int y = 0; y < image.rows; y++){
            cv::Vec3b intensity = colorSegHSV.at<cv::Vec3b>(y, x);

            if (intensity[2] > maxV * 0.7){
                final.at<uchar>(y, x) = 255;
            }
            else {
                final.at<uchar>(y, x) = 0;
            }
        }
    }*/


    cv::imshow("Output", colorSeg);
    cv::waitKey(40000);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    findContours(final.clone(), contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);

    QVector<cv::Rect> filteredRects;
    QVector<std::vector<cv::Point> > filteredContours;

    for (unsigned int i = 0; i < contours.size(); i++){
        cv::Rect rect = minAreaRect(contours[i]).boundingRect();
        double ratio = (double)rect.width / rect.height;
        if (rect.width > 10 && rect.width < 40 && rect.height > 15 && rect.height < image.rows * 0.66){

            if (rect.x < 0){
                rect.x = 0;
            }
            if (rect.y < 0){
                rect.y = 0;
            }

            if (rect.x + rect.width > final.cols){
                rect.width = final.cols - rect.x;
            }
            if (rect.y + rect.height > final.rows){
                rect.height = final.rows - rect.y;
            }

            filteredRects.push_back(rect);
            filteredContours.push_back(contours[i]);
        }
    }

    QVector<cv::Rect> sortedRects;
    QVector<std::vector<cv::Point> > sortedContours;

    for (int i = 0; i < filteredRects.count(); i++){
        double min = image.cols;
        int index = -1;
        for (int j = 0; j < filteredRects.count(); j++){
            if (filteredRects[j].x < min){
                min = filteredRects[j].x;
                index = j;
            }
        }

        if (index != -1){
            sortedRects.push_back(cv::Rect(filteredRects[index]));
            sortedContours.push_back(filteredContours[index]);
            filteredRects[index].x = image.cols;
        }
        else {
            qDebug() << "Error in sort, some value must be bigger than image.cols";
        }
    }

    QVector<cv::Rect> finalRects;
    QVector<std::vector<cv::Point> > finalContours;

    if (sortedContours.size() > 2){
        finalRects.push_back(sortedRects[sortedContours.size()/2]);
        finalContours.push_back(sortedContours[sortedContours.size()/2]);

        finalRects.push_back(sortedRects[sortedContours.size()/2 + 1]);
        finalContours.push_back(sortedContours[sortedContours.size()/2 + 1]);
    }
    else {
        finalRects = sortedRects;
        finalContours = sortedContours;
    }

    if (finalContours.size() == 0){
        return -1;
    }
    else if (finalContours.size() == 1){
        return digitHelper(final, digitsOnField, finalContours[0], finalRects[0]);
    }
    else {
        int digit1 = digitHelper(final, digitsOnField, finalContours[0], finalRects[0]);
        int digit2 = digitHelper(final, digitsOnField, finalContours[1], finalRects[1]);

        if (digit1 == 1){
            return 10 + digit2;
        }
        else {
            // temporary
            return digit2;
        }
    }
}

int digitHelper(cv::Mat bigImage, QList<int> digitsOnField, std::vector<cv::Point> contour, cv::Rect rect) {
    cv::Mat digitImage(rect.height, rect.width, CV_8U);
    for (int x = rect.x; x < rect.x + rect.width; x++){
        for (int y = rect.y; y < rect.y + rect.height; y++){
            if (bigImage.at<uchar>(y, x) == 255){
                if (!(pointPolygonTest(contour, cv::Point2f(x, y), true) >= 0)){
                    digitImage.at<uchar>(y-rect.y, x-rect.x) = 0;
                }
                else {
                    digitImage.at<uchar>(y-rect.y, x-rect.x) = bigImage.at<uchar>(y,x);
                }
            }
            else {
                digitImage.at<uchar>(y-rect.y, x-rect.x) = 0;
            }
        }
    }

    //qDebug() << digitImage.cols << digitImage.rows;
    //cv::imshow("Output", digitImage);
    //cv::waitKey(40000);

    std::vector<cv::Point> shiftedContour = contour;

    for (int j = 0; j < shiftedContour.size(); j++){
        shiftedContour[j].x -= rect.x;
        shiftedContour[j].y -= rect.y;
    }

    return mostProbableDigit(digitImage, digitsOnField, shiftedContour);
}

int mostProbableDigit(cv::Mat numberImage, QList<int> digitsOnField, std::vector<cv::Point> contour){

    cv::RotatedRect rect = cv::minAreaRect(contour);

    float angle = rect.angle;

    cv::Size rectSize = rect.size;
    if (rect.angle <= -45) {
        angle += 90.0;
        cv::swap(rectSize.width, rectSize.height);
    }

    cv::Mat rotationMatrix = getRotationMatrix2D(rect.center, angle, 1.0);
    cv::Mat rotated = numberImage.clone();
    cv::Mat cropped = numberImage.clone();

    warpAffine(numberImage, rotated, rotationMatrix, cv::Size(numberImage.size().width, 2*numberImage.size().height), cv::INTER_LANCZOS4);
    getRectSubPix(rotated, rectSize, rect.center, cropped);

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(36, 45));

    cv::Mat skeleton = thinningGuoHall(resized);
    Skeleton ske(skeleton, resized);

    QVector<int> possibleDigits = ske.possibleNumbers(digitsOnField).toVector();

    if (possibleDigits.empty()){
        return 0;
    }
    else {
        return possibleDigits[0];
    }
}

void runOnDataSet(QList<int> digitsOnField){
    int success = 0;
    int fail = 0;

    for (int i = 0; i < 13; i++){
        if (i != 0 && i != 1 && i != 2 && i != 3 && i != 4 && i != 7 && i != 10 && i != 11){
            for (int j = 0; j <= 9; j++){
                cv::Mat image = cv::imread(QString("temp/dataset/" + QString::number(i) + "/" + QString::number(j) + ".png").toStdString());

                qDebug() << i << j;

                int num = mostProbableNumber(image, digitsOnField);
                if (num == i){
                    success++;
                    qDebug() << "Success !" << num;
                }
                else {
                    fail++;
                    qDebug() << "Failure !" << num;
                }

                image.release();
            }
        }
    }

    qDebug() << "Total Success : " << success << " , " << (success * 100) / (success + fail) << " %";
    qDebug() << "Total Failure : " << fail << " , " << (fail * 100) / (success + fail) << " %";
}
