#include "functions.h"
#include "configuration.h"
#include <QDebug>

#define BIN_SIZE 20

NumPos mostProbableNumber(cv::Mat image, QList<int> digitsOnField){
    cv::Mat blurredImage;
    cv::Mat labImage;
    cv::Mat final;
    cv::Mat jerseyFinal;
    GaussianBlur(image, blurredImage, cv::Size(1, 1), 0, 0);
    cvtColor(blurredImage, labImage, CV_BGR2Lab);
    cvtColor(image, jerseyFinal, CV_BGR2GRAY);
    cvtColor(image, final, CV_BGR2GRAY);

    cv::Mat colorSeg;
    cv::Mat colorSegHSV;
    pyrMeanShiftFiltering(labImage, colorSegHSV, 20, 20, 1); // change name later
    cvtColor(colorSegHSV, colorSeg, CV_Lab2BGR);

    int histo[255/BIN_SIZE][255/BIN_SIZE][255/BIN_SIZE];
    int widerHisto[255/BIN_SIZE][255/BIN_SIZE][255/BIN_SIZE];

    for (int i = 0; i < 255 / BIN_SIZE; i++){
        for (int j = 0; j < 255 / BIN_SIZE; j++){
            for (int k = 0; k < 255 / BIN_SIZE; k++){
                histo[i][j][k] = 0;
                widerHisto[i][j][k] = 0;
            }
        }
    }

    long meanV = 0;
    int maxL = 0;
    for (int x = 0; x < image.cols; x++){
        for (int y = 0; y < image.rows; y++){
            cv::Vec3b val = colorSegHSV.at<cv::Vec3b>(y, x);

            if (val[0]/BIN_SIZE != 0 || val[1]/BIN_SIZE != 128/BIN_SIZE || val[2]/BIN_SIZE != 128/BIN_SIZE){ // black pixels dont count
                histo[val[0]/BIN_SIZE][val[1]/BIN_SIZE][val[2]/BIN_SIZE]++;

                meanV += val[2];

                if (maxL < val[0]){
                    maxL = val[0];
                }

                //qDebug() << "Temp : " << val[0] << val[1] << val[2];
            }

        }
    }

    meanV /= (image.cols * image.rows);

    for (int i = 1; i < 255 / BIN_SIZE - 1; i++){
        for (int j = 1; j < 255 / BIN_SIZE - 1; j++){
            for (int k = 1; k < 255 / BIN_SIZE - 1; k++){

                for (int i2 = -1; i2 <= 1; i2++){
                    for (int j2 = -1; j2 <= 1; j2++){
                        for (int k2 = -1; k2 <= 1; k2++){
                            widerHisto[i][j][k] += histo[i + i2][j + j2][k + k2];
                        }
                    }
                }
            }
        }
    }

    int maxH = 0;
    int maxS = 0;
    int maxV = 0;
    int maxVote = 0;
    for (int i = 0; i < 255 / BIN_SIZE; i++){
        for (int j = 0; j < 255 / BIN_SIZE; j++){
            for (int k = 0; k < 255 / BIN_SIZE; k++){
                if (histo[i][j][k] > maxVote){
                    maxVote = histo[i][j][k];
                    maxH = i * BIN_SIZE + BIN_SIZE / 2;
                    maxS = j * BIN_SIZE + BIN_SIZE / 2;
                    maxV = k * BIN_SIZE + BIN_SIZE / 2;

//                    qDebug() << "Vote : " << maxVote;
//                    qDebug() << "Temp : " << maxH * (BIN_SIZE * 360 / 255) << maxS * (BIN_SIZE * 100 / 255) << maxV * (BIN_SIZE * 100 / 255);
                }
            }
        }
    }

    qDebug() << "Values : " << maxH * 100 / 255 << maxS - 128 << maxV - 128;
    //qDebug() << "Indices : " << (maxH - BIN_SIZE / 2) / BIN_SIZE << (maxS - BIN_SIZE / 2) / BIN_SIZE << (maxV - BIN_SIZE / 2) / BIN_SIZE;

    double th = 1;
    double ts = 1;
    double tv = 1;

    if (maxH > 255 - th * BIN_SIZE){
        maxH -= 255;
    }

    for (int x = 0; x < image.cols; x++){
        for (int y = 0; y < image.rows; y++){
            cv::Vec3b val = colorSegHSV.at<cv::Vec3b>(y, x);

            if (((val[0] > maxH - th * BIN_SIZE && val[0] < maxH + th * BIN_SIZE) || (val[0] - 255 > maxH - th * BIN_SIZE && val[0] - 255 < maxH + th * BIN_SIZE)) && val[1] > maxS - ts * BIN_SIZE && val[1] < maxS + ts * BIN_SIZE && val[2] > maxV - tv * BIN_SIZE && val[2] < maxV + tv * BIN_SIZE){
                jerseyFinal.at<uchar>(y, x) = 255;
            }
            else {
                jerseyFinal.at<uchar>(y, x) = 0;
            }
        }
    }

    std::vector<std::vector<cv::Point> > jerseyContours;
    std::vector<cv::Vec4i> jerseyHierarchy;

    cv::Rect jerseyRect;

    findContours(jerseyFinal.clone(), jerseyContours, jerseyHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    if (jerseyContours.size() < 1){
        qDebug() << "Error, 0 shirt detected";

        for (int x = 0; x < jerseyFinal.cols; x++){
            for (int y = 0; y < jerseyFinal.rows; y++){
                final.at<uchar>(y, x) = 0;
            }
        }
    }
    else {
        std::vector<cv::Point> trueJerseyContour = jerseyContours[0];

        for (int i = 0; i < jerseyContours.size(); i++){
            if (minAreaRect(jerseyContours[i]).boundingRect().height > minAreaRect(trueJerseyContour).boundingRect().height){
                trueJerseyContour = jerseyContours[i];
            }
        }

        std::vector<cv::Point> approximateContour;

        cv::approxPolyDP(cv::Mat(trueJerseyContour), approximateContour, 3, true );

        std::vector<cv::Vec4i> defect;
        std::vector<int> hull;

        std::vector<cv::Point> convexContour;

        cv::convexHull(trueJerseyContour, convexContour, false, true);
        //cv::convexityDefects(trueJerseyContour, convexHull, defect);

        jerseyRect = cv::minAreaRect(convexContour).boundingRect();

        std::vector<cv::Point> convexJerseyContour;
        for (int i = 0; i < defect.size(); i++){
            convexJerseyContour.push_back(trueJerseyContour[defect[i][0]]);
            convexJerseyContour.push_back(trueJerseyContour[defect[i][1]]);
        }

        if (convexJerseyContour.size() == 0){
            convexJerseyContour = trueJerseyContour;
        }
        else {
            convexJerseyContour.push_back(trueJerseyContour[defect[defect.size()-1][1]]);
        }

        std::vector<std::vector<cv::Point> > temp;
        temp.push_back(convexContour);

        for (int x = 0; x < jerseyFinal.cols; x++){
            for (int y = 0; y < jerseyFinal.rows; y++){
                if (jerseyFinal.at<uchar>(y, x) == 255){
                    final.at<uchar>(y, x) = 0;
                }
                else {
                    cv::Vec3b val = colorSegHSV.at<cv::Vec3b>(y, x);

                    if (pointPolygonTest(convexContour, cv::Point2f(x,y), true) >= 4 && val[0] >= maxH){
                        final.at<uchar>(y, x) = 255;
                    }
                    else {
                        final.at<uchar>(y, x) = 0;
                    }
                }
            }
        }

        cv::Scalar color = cv::Scalar(0, 0, 255);
        cv::drawContours(colorSeg, temp, 0, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
    }

    /*for (int x = 0; x < image.cols; x++){
        for (int y = 0; y < image.rows; y++){
            cv::Vec3b intensity = image.at<cv::Vec3b>(y, x);

            float sampleData[3];
            sampleData[0] = intensity[0];
            sampleData[1] = intensity[1];
            sampleData[2] = intensity[2];

            cv::Mat sampleMat(1, 3, CV_32FC1, sampleData);

            float responseGreen = Configuration::jerseyMachines.m[1]->predict(sampleMat);

            if (responseGreen == 1.0){
                final.at<uchar>(y, x) = 255;
            }
            else {
                /*float responseRed = Configuration::jerseyMachines.m[1]->predict(sampleMat);

                if (responseRed == 1.0){
                    final.at<uchar>(y, x) = 255;
                }
                else {

                }*/

                /*final.at<uchar>(y, x) = 0;
            }
        }
    }*/

    /*cv::Mat colorSeg;
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
    }*/

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


//    cv::imshow("Output", colorSeg);
//    cv::waitKey(40000);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    findContours(final.clone(), contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);

    QVector<cv::Rect> filteredRects;
    QVector<std::vector<cv::Point> > filteredContours;

    for (unsigned int i = 0; i < contours.size(); i++){
        cv::Rect rect = minAreaRect(contours[i]).boundingRect();
        double ratio = (double)rect.width / rect.height;
        if (rect.width > jerseyRect.width * 0.25 && rect.width < jerseyRect.width * 0.9 && rect.height > jerseyRect.height * 0.3 && rect.height < jerseyRect.height * 0.8 && ratio > 0.5 && ratio < 1.5){

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

    NumPos result;

    if (finalContours.size() == 0){
        result.number = -1;
        return result;
    }
    else if (finalContours.size() == 1){
        result.number = digitHelper(final, digitsOnField, finalContours[0], finalRects[0]);
        result.pos.x = finalRects[0].x + finalRects[0].width/2;
        result.pos.y = finalRects[0].y + finalRects[0].height/2;
        return result;
    }
    else {
        int digit1 = digitHelper(final, digitsOnField, finalContours[0], finalRects[0]);
        int digit2 = digitHelper(final, digitsOnField, finalContours[1], finalRects[1]);

        if (digit1 == 1){
            result.number = 10 + digit2;
            result.pos.x = finalRects[0].x + finalRects[0].width/2;
            result.pos.y = finalRects[0].y + finalRects[0].height/2;
            return result;
        }
        else {
            // we return the 2nd digit

            result.number = digit2;
            result.pos.x = finalRects[1].x + finalRects[1].width/2;
            result.pos.y = finalRects[1].y + finalRects[1].height/2;
            return result;
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

//    cv::imshow("Output", resized);
//    cv::waitKey(40000);

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
    int notFound = 0;

    for (int i = 0; i < 12; i++){
        if (i != 0 && i != 1 && i != 2 && i != 3 && i != 4 && i != 7 && i != 10 && i != 11){
            for (int j = 0; j <= 9; j++){
                cv::Mat image = cv::imread(QString("temp/dataset/" + QString::number(i) + "/" + QString::number(j) + ".png").toStdString());

                qDebug() << i << j;

                int num = mostProbableNumber(image, digitsOnField).number;
                if (num == i){
                    success++;
                    qDebug() << "Success !" << num;
                }
                else if (num == -1){
                    notFound++;
                    qDebug() << "Not found !" << num;
                }
                else {
                    fail++;
                    qDebug() << "Failure !" << num;
                }

                image.release();
            }
        }
    }

    qDebug() << "Total Success : " << success << " , " << (success * 100) / (success + fail + notFound) << " %";
    qDebug() << "Total Failure : " << fail << " , " << (fail * 100) / (success + fail + notFound) << " %";
    qDebug() << "Total Not Found : " << notFound << " , " << (notFound * 100) / (success + fail + notFound) << " %";
}
