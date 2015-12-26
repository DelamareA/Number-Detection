#include <QFile>
#include <QTextStream>
#include <QDebug>
#include "headers.h"

using namespace cv;

int loadAndRun();

int main(int argc, char *argv[]){
    // Below is the code to generate the datasets to train the svms

    /*QList<int> all;
    for (int i = 0; i < 10; i++){
        all.push_back(i);
    }

    for (int i = 0; i < 10; i++){
        for (int j = i+1; j < 10; j++){
            qDebug() << "Generating SVM " << i << "-" << j;
            QList<int> numbers;
            numbers.push_back(i);
            numbers.push_back(j);
            generateDataSet(numbers, 10, 36, 45, "svm/" + QString::number(i) + "-" + QString::number(j) + "/");
            generateSVM("svm/" + QString::number(i) + "-" + QString::number(j) + "/", M0);
        }
    }*/

    return loadAndRun();
}

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
        outputVideo.open(Config::getOutputVideoPath().toStdString(), -1, 1, size, true);

        if (!outputVideo.isOpened()){
            qDebug() << "Could not open video output";
            return -1;
        }

        cv::Mat image;
        inputVideo >> image;
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

            out = frameProcess(image, background);
            outputVideo << out->getImage();
            frameCount++;
            outputText += out->toString();

            delete out;

            qDebug() << "End frame : " << frameCount-1;

            for (int i = 0; i < inputVideo.get(CV_CAP_PROP_FPS)/4; i++){
                inputVideo >> image;
                count++;
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

        out = frameProcess(image, background);

        cv::namedWindow("Output");
        cv::imshow("Output", out->getImage());
        cv::waitKey(40000);

        delete out;
    }

    return 0;
}
