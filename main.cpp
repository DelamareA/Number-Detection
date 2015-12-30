#include <QDebug>
#include "headers.h"

using namespace cv;

int loadAndRun();

int main(){
    // Below is the code to generate the datasets to train the svms

    for (int i = 0; i < 10; i++){
        qDebug() << "Generating SVM " << i << "- all";
        generateSVM("svm/" + QString::number(i) + "-all/", i, -1, 0);
        for (int j = i+1; j < 10; j++){
            qDebug() << "Generating SVM " << i << "-" << j;
            generateSVM("svm/" + QString::number(i) + "-" + QString::number(j) + "/", i, j, 1);
        }
    }

    return loadAndRun();
}

/**
 * @brief loadAndRun Main function that loads the config file, the images, the video and launches the frameProcess function.
 * @return 0 iff the program terminates normally.
 */
int loadAndRun(){
    Config::setConfigFromFile("config.txt");
    cv::Mat background = cv::imread(Config::getBackgroundPath().toStdString());

    Output* out = 0;

//    runOnDataSet();

    if (Config::getIsVideo()){
        cv::VideoCapture inputVideo(Config::getVideoPath().toStdString());
        if (!inputVideo.isOpened()){
            qDebug() << "Could not open video";
            return -1;
        }

        cv::Size size = cv::Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH), (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));

        cv::VideoWriter outputVideo;
        outputVideo.open(Config::getOutputVideoPath().toStdString(), -1, inputVideo.get(CV_CAP_PROP_FPS), size, true);

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

            imwrite(QString("tempframes/" + QString::number(frameCount)+  ".png").toStdString(), image);

            if (count < 20 || !Config::getIsMogUsed()){
                out = frameProcess(image, background, foregroundMask, NORMAL);
            }
            else {
                out = frameProcess(image, background, foregroundMask, MOG);
            }

            outputVideo << out->getImage();
            frameCount++;
            outputText += out->toString();

            delete out;

            qDebug() << "End frame : " << frameCount-1;

            for (int i = 0; i < inputVideo.get(CV_CAP_PROP_FPS)/inputVideo.get(CV_CAP_PROP_FPS); i++){
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
