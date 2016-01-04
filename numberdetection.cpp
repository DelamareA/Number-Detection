#include <QDebug>
#include "headers.h"
#include "config.h"

#define BIN_SIZE 20

/**
 * @brief mostProbableNumber Function, that given a player's image, outputs the number.
 * @param image The player's image.
 * @return The number or -1 if none was detected.
 */
NumPos mostProbableNumber(cv::Mat image){
    cv::Mat labImage;
    cvtColor(image, labImage, CV_BGR2Lab);

    cv::Mat colorSeg;
    cv::Mat colorSegLab;
    pyrMeanShiftFiltering(labImage, colorSegLab, 18, 18, 1);
    cvtColor(colorSegLab, colorSeg, CV_Lab2BGR);

    // Detection of the jersey (which color is the dominant color of the image)

    int histo[255/BIN_SIZE][255/BIN_SIZE][255/BIN_SIZE];

    for (int i = 0; i < 255 / BIN_SIZE; i++){
        for (int j = 0; j < 255 / BIN_SIZE; j++){
            for (int k = 0; k < 255 / BIN_SIZE; k++){
                histo[i][j][k] = 0;
            }
        }
    }

    for (int x = 0; x < image.cols; x++){
        for (int y = 0; y < image.rows; y++){
            cv::Vec3b val = colorSegLab.at<cv::Vec3b>(y, x);

            if (val[0]/BIN_SIZE != 0 || val[1]/BIN_SIZE != 128/BIN_SIZE || val[2]/BIN_SIZE != 128/BIN_SIZE){ // black pixels don't count
                histo[val[0]/BIN_SIZE][val[1]/BIN_SIZE][val[2]/BIN_SIZE]++;
            }

        }
    }

    int maxVoteL = 0;
    int maxVoteA= 0;
    int maxVoteB = 0;
    int maxVote = 0;
    for (int i = 0; i < 255 / BIN_SIZE; i++){
        for (int j = 0; j < 255 / BIN_SIZE; j++){
            for (int k = 0; k < 255 / BIN_SIZE; k++){
                if (histo[i][j][k] > maxVote){
                    maxVote = histo[i][j][k];
                    maxVoteL = i * BIN_SIZE + BIN_SIZE / 2;
                    maxVoteA = j * BIN_SIZE + BIN_SIZE / 2;
                    maxVoteB = k * BIN_SIZE + BIN_SIZE / 2;
                }
            }
        }
    }

    //qDebug() << "Value of the dominant color : " << maxVoteL * 100 / 255 << maxVoteA - 128 << maxVoteB - 128;

    cv::Mat jerseyFinal; // the image with the jersey appearing in white
    cvtColor(image, jerseyFinal, CV_BGR2GRAY);

    double tL = 26;
    double tA = 20;
    double tB = 20;

    for (int x = 0; x < image.cols; x++){
        for (int y = 0; y < image.rows; y++){
            cv::Vec3b val = colorSegLab.at<cv::Vec3b>(y, x);

            if (abs(val[0] - maxVoteL) < tL && abs(val[1] - maxVoteA) < tA && abs(val[2] - maxVoteB) < tB){
                jerseyFinal.at<uchar>(y, x) = 255;
            }
            else {
                jerseyFinal.at<uchar>(y, x) = 0;
            }
        }
    }

    cv::Mat jerseyFinalEroded;
    int dilateSize = 1;
    cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*dilateSize + 1, 2*dilateSize + 1), cv::Point(dilateSize, dilateSize) );
    cv::erode(jerseyFinal, jerseyFinalEroded, dilateElement);

    std::vector<std::vector<cv::Point> > jerseyContours;
    std::vector<cv::Vec4i> jerseyHierarchy;

    cv::Rect jerseyRect;

    findContours(jerseyFinalEroded, jerseyContours, jerseyHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    cv::Mat final; // the image with the number in white
    cvtColor(image, final, CV_BGR2GRAY);

    if (jerseyContours.size() < 1){
        qDebug() << "Error, 0 shirt detected";

        for (int x = 0; x < jerseyFinal.cols; x++){
            for (int y = 0; y < jerseyFinal.rows; y++){
                final.at<uchar>(y, x) = 0;
            }
        }
    }
    else {
        // We find the contour of the jersey and we find the number in it

        std::vector<cv::Point> trueJerseyContour = jerseyContours[0];

        for (unsigned int i = 0; i < jerseyContours.size(); i++){
            if (minAreaRect(jerseyContours[i]).boundingRect().area() > minAreaRect(trueJerseyContour).boundingRect().area()){
                trueJerseyContour = jerseyContours[i];
            }
        }

        bool isJerseyValid = (minAreaRect(trueJerseyContour).boundingRect().height > image.rows * 0.7);

        std::vector<cv::Point> convexContour;

        cv::convexHull(trueJerseyContour, convexContour, false, true);

        jerseyRect = cv::minAreaRect(convexContour).boundingRect();

        for (int x = 0; x < jerseyFinal.cols; x++){
            for (int y = 0; y < jerseyFinal.rows; y++){
                if (jerseyFinal.at<uchar>(y, x) == 255 || !isJerseyValid){
                    final.at<uchar>(y, x) = 0;
                }
                else {
                    cv::Vec3b val = colorSegLab.at<cv::Vec3b>(y, x);

                    if (pointPolygonTest(convexContour, cv::Point2f(x,y), true) >= 0
                            && pointPolygonTest(trueJerseyContour, cv::Point2f(x,y), true) >= -8
                            && val[0] >= maxVoteL){
                        final.at<uchar>(y, x) = 255;
                    }
                    else {
                        final.at<uchar>(y, x) = 0;
                    }
                }
            }
        }

        // Code to show the contour
        std::vector<std::vector<cv::Point> > temp;
        temp.push_back(convexContour);

        cv::Scalar color = cv::Scalar(0, 0, 255);
        cv::drawContours(colorSeg, temp, 0, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
    }

//    cv::imshow("Output", colorSeg);
//    cv::waitKey(40000);

//    cv::imshow("Output", final);
//    cv::waitKey(40000);

    // Number detection

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    findContours(final.clone(), contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);

    QVector<cv::Rect> filteredRects;
    QVector<std::vector<cv::Point> > filteredContours;

    QVector<cv::Rect> filteredRectsMaybe;
    QVector<std::vector<cv::Point> > filteredContoursMaybe;

    for (unsigned int i = 0; i < contours.size(); i++){
        cv::Rect rect = minAreaRect(contours[i]).boundingRect();
        cv::RotatedRect rotatedRect = minAreaRect(contours[i]);

        if (rotatedRect.size.width > rotatedRect.size.height){
            cv::swap(rotatedRect.size.width, rotatedRect.size.height);
        }
        double ratio = (double)rotatedRect.size.width / rotatedRect.size.height;
        if (rotatedRect.size.width > jerseyRect.width * 0.18
                && rotatedRect.size.width < jerseyRect.width * 0.9
                && rotatedRect.size.height > jerseyRect.height * 0.25
                && rotatedRect.size.height < jerseyRect.height * 0.8
                && ratio > 0.4
                && contourArea(contours[i]) > image.rows * image.rows * 0.015){

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

        if (rotatedRect.size.width > jerseyRect.width * 0.12
                && rotatedRect.size.width < jerseyRect.width * 0.9
                && rotatedRect.size.height > jerseyRect.height * 0.25
                && rotatedRect.size.height < jerseyRect.height * 0.8
                && ratio > 0.325
                && contourArea(contours[i]) > image.rows * image.rows * 0.01){

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

            filteredRectsMaybe.push_back(rect);
            filteredContoursMaybe.push_back(contours[i]);
        }
    }

    if (filteredRects.size() == 1){
        int biggestMaybe = -1;
        int maxArea = 0;

        for (int i = 0; i < filteredContoursMaybe.size(); i++){
            if (contourArea(filteredContoursMaybe[i]) > maxArea){
                maxArea = contourArea(filteredContoursMaybe[i]);
                biggestMaybe = -1;
            }
        }

        if (biggestMaybe != -1 && filteredRectsMaybe[biggestMaybe].x < filteredRects[0].x){
            filteredRects.push_back(filteredRectsMaybe[biggestMaybe]);
            filteredContoursMaybe.push_back(filteredContours[biggestMaybe]);
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

    // Selection of at most 2 contours
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
        result.number = digitHelper(final, finalContours[0], finalRects[0]);
        result.pos.x = finalRects[0].x + finalRects[0].width/2;
        result.pos.y = finalRects[0].y + finalRects[0].height/2;
        return result;
    }
    else {
        int digit1 = digitHelper(final, finalContours[0], finalRects[0]);
        int digit2 = digitHelper(final, finalContours[1], finalRects[1]);

        if (Config::getNumbersOnField().contains(digit1 * 10 + digit2)){
            result.number = digit1 * 10 + digit2;
            result.pos.x = finalRects[0].x + finalRects[0].width/2;
            result.pos.y = finalRects[0].y + finalRects[0].height/2;
            return result;
        }
        else if (Config::getNumbersOnField().contains(digit2)){
            result.number = digit2;
            result.pos.x = finalRects[1].x + finalRects[1].width/2;
            result.pos.y = finalRects[1].y + finalRects[1].height/2;
            return result;
        }
        else {
            result.number = digit1;
            result.pos.x = finalRects[0].x + finalRects[0].width/2;
            result.pos.y = finalRects[0].y + finalRects[0].height/2;
            return result;
        }
    }
}

/**
 * @brief digitHelper Helper functions which crops the image.
 * @param bigImage The wider image of the player.
 * @param contour The contour of the digit to crop.
 * @param rect The minimum rect which includes the number.
 * @return The digit value corresponding to the image.
 */
int digitHelper(cv::Mat bigImage, std::vector<cv::Point> contour, cv::Rect rect) {
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

    std::vector<cv::Point> shiftedContour = contour;

    for (unsigned int j = 0; j < shiftedContour.size(); j++){
        shiftedContour[j].x -= rect.x;
        shiftedContour[j].y -= rect.y;
    }

    return mostProbableDigit(digitImage, shiftedContour);
}

/**
 * @brief mostProbableDigit Outputs the most probable digit on an image.
 * @param digitImage The image of the digit, in black and white.
 * @param contour The contour of the digit.
 * @return the most probable digit on an image.
 */
int mostProbableDigit(cv::Mat digitImage, std::vector<cv::Point> contour){
    cv::RotatedRect rect = cv::minAreaRect(contour);

    static int count = 0;

    float angle = rect.angle;

    cv::Size rectSize = rect.size;
    if (rect.angle <= -45) {
        angle += 90.0;
        cv::swap(rectSize.width, rectSize.height);
    }

    cv::Mat rotationMatrix = getRotationMatrix2D(rect.center, angle, 1.0);
    cv::Mat rotated = digitImage.clone();
    cv::Mat cropped = digitImage.clone();

    warpAffine(digitImage, rotated, rotationMatrix, cv::Size(digitImage.size().width, 2*digitImage.size().height), cv::INTER_LANCZOS4);
    getRectSubPix(rotated, rectSize, rect.center, cropped);

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(36, 45));

    imwrite(QString("digit-dataset/" + QString::number(count) + ".png").toStdString(), resized);

    Skeleton ske(resized);

    int result = ske.mostProbableDigit();

    //qDebug() << count++ << " -> " << result;

    return result;
}

/**
 * @brief runOnDataSet Function that runs a set of images to test the number detection part.
 */
void runOnDataSet(){
    int success = 0;
    int fail = 0;
    int notFound = 0;

    for (int i = 0; i < 12; i++){
        if (i != 0 && i != 1 && i != 2 && i != 3 && i != 4 && i != 7 && i != 10 && i != 11){
            for (int j = 0; j <= 9; j++){
                cv::Mat image = cv::imread(QString("temp/dataset/" + QString::number(i) + "/" + QString::number(j) + ".png").toStdString());

                qDebug() << i << j;

                int num = mostProbableNumber(image).number;
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
