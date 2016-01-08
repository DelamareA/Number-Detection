#include <QDebug>
#include "headers.h"

using namespace cv;

int loadAndRun();

int main(){
    // Below is the code to generate the datasets to train the svms

    /*for (int i = 0; i < 10; i++){
        qDebug() << "Generating SVM " << i << "- all";
        generateSVM("svm/" + QString::number(i) + "-all/", i, -1, 0);
        for (int j = i+1; j < 10; j++){
            qDebug() << "Generating SVM " << i << "-" << j;
            generateSVM("svm/" + QString::number(i) + "-" + QString::number(j) + "/", i, j, 1);
        }
    }*/

    return loadAndRun();
}

/**
 * @brief loadAndRun Main function that loads the config file, the images, the video and launches the frameProcess function.
 * @return 0 iff the program terminates normally.
 */
int loadAndRun(){
    Config::setConfigFromFile("config.txt");
    cv::Mat background = cv::imread(Config::getBackgroundPath().toStdString());

    FrameOutput* out = 0;
    FrameOutput* oldOut = 0;
    FrameOutput* oldestOut = 0;

    if (Config::getIsVideo()){
        cv::VideoCapture inputVideo(Config::getVideoPath().toStdString());
        if (!inputVideo.isOpened()){
            qDebug() << "Could not open video";
            return -1;
        }

        cv::Size size = cv::Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH), (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

        cv::VideoWriter outputVideo;
        outputVideo.open(Config::getOutputVideoPath().toStdString(), -1, 1, size, true);

        if (!outputVideo.isOpened()){
            qDebug() << "Could not open video output";
            return -1;
        }

        cv::Ptr<cv::BackgroundSubtractor> bgs = cv::createBackgroundSubtractorMOG2();
        cv::Mat foregroundMask;

        cv::Mat image;
        inputVideo >> image;
        bgs->apply(image, foregroundMask);
        int count = 0;
        int frameCount = 0;

        QString outputText;
        int width = 0;
        int height = 0;

        while (!image.empty()){
            if (image.rows != background.rows || image.cols != background.cols){
                qDebug() << "Image and background have not the same size : " << image.cols << "x" << image.rows;
            }

            width = image.cols;
            height = image.rows;

            qDebug() << "Start frame : " << frameCount;

            if (Config::getIsPostProcessActivated()){
                if (oldestOut != 0){
                    delete oldestOut;
                }

                oldestOut = oldOut;
                oldOut = out;
            }

            if (count < 20 || !Config::getIsMogUsed()){
                out = frameProcess(image, background, foregroundMask, NORMAL);
            }
            else {
                out = frameProcess(image, background, foregroundMask, MOG);
            }

            if (Config::getIsPostProcessActivated()){
                if (oldestOut != 0 && oldOut != 0){
                    QList<cv::Point2i> datas = oldOut->getAllData();

                    for (int j = 0; j < datas.size(); j++){
                        if (!oldestOut->isDataClose(datas[j].x, datas[j].y, 50) && !out->isDataClose(datas[j].x, datas[j].y, 50)){
                            oldOut->removeData(datas[j].x, datas[j].y, 1);
                            qDebug() << "Data removed at " << datas[j].x << datas[j].y;
                        }
                    }

                    outputVideo << oldOut->getImage();
                    outputText += oldOut->toString();
                }
                else if (oldOut == 0){ // first frame

                    outputVideo << out->getImage();
                    outputText += out->toString();
                }
            }
            else {
                outputVideo << out->getImage();
                outputText += out->toString();

                delete out;
            }

            frameCount++;

            qDebug() << "End frame : " << frameCount-1;

            for (int i = 0; i < inputVideo.get(CV_CAP_PROP_FPS)/4; i++){
                inputVideo >> image;
                if (!image.empty() && Config::getIsMogUsed()){
                    bgs->apply(image, foregroundMask);
                }
                count++;

                if (count % 2000 == 0 && Config::getIsMogUsed()){

                    // each 2000 frames, create a new background
                    bgs->getBackgroundImage(background);
                }
            }
        }

        if (Config::getIsPostProcessActivated()){
            outputVideo << out->getImage();
            outputText += out->toString();
        }

        outputText = QString::number(width) + '@' + QString::number(height) + '@' + QString::number(frameCount) + "@" + outputText + "@";

        QFile file(Config::getOutputTextPath());
        if (file.open(QIODevice::WriteOnly)){
            QTextStream stream(&file);
            stream << outputText << endl;
            file.close();
        }
        else {
            qDebug() << "Cannot open " + Config::getOutputTextPath();
        }

    }
    else {
        cv::Mat image = cv::imread(Config::getImagePath().toStdString());

        if (image.rows != background.rows || image.cols != background.cols){
            qDebug() << "Image and background have not the same size";
        }

        out = frameProcess(image, background, cv::Mat(), NORMAL);

        cv::namedWindow("Output");
        cv::imshow("Output", out->getImage());
        cv::waitKey(40000);

        delete out;
    }

    return 0;
}
