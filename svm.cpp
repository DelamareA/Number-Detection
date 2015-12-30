#include <QList>
#include <QPixmap>
#include <QPainter>
#include <QTime>
#include <QApplication>
#include <QDebug>
#include <QColor>
#include <QFile>
#include <qglobal.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "headers.h"

using namespace cv;
using namespace cv::ml;

/**
 * @brief generateDataSet Generates a dataset of digit images.
 * @param numbers The different numbers to generate.
 * @param countPerNumber The number of images to create per number.
 * @param width The width of the images.
 * @param height The height of the images.
 * @param outputPath Where the images have to be written.
 */
void generateDataSet(QList<int> numbers, int countPerNumber, int width, int height, QString outputPath) {
    char **argv = new char*[1];
    argv[0] = new char[1];
    int argc = 1;
    QApplication a(argc, argv); // just to use QPixmaps

    qsrand(QTime::currentTime().msec());

    int size = 50;

    QString labels = QString::number(numbers.size() * countPerNumber);

    for (int i = 0; i < numbers.size(); i++){
        for (int j = 0; j < countPerNumber; j++){
            QPixmap pix(3 * width, 3 * height);
            pix.fill(Qt::black);
            QPainter painter(&pix);
            painter.setPen(QColor(255, 255, 255));

            QFont font("Arial", size);
            font.setBold(true);
            painter.setFont(font);

            painter.translate(width, size);
            double realAngle = (rand() % 10) - 5;
            painter.rotate(realAngle);
            painter.drawText(QPoint(0, size), QString::number(numbers[i]));

            QImage image = pix.toImage();

            Mat mat = Mat::zeros(pix.height(), pix.width(), CV_8U);
            for (int x = 0; x < pix.width(); x++){
                for (int y = 0; y < pix.height(); y++){
                    QRgb val = image.pixel(x, y);

                    if (qRed(val) > 120){
                        mat.at<uchar>(y, x) = 255;
                    }
                    else {
                        mat.at<uchar>(y, x) = 0;
                    }
                }
            }

            // GET CONTOUR

            std::vector<std::vector<Point> > contours;
            std::vector<Vec4i> hierarchy;

            findContours(mat.clone(), contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE);

            if (contours.size() < 1){
                qDebug() << "Error : 0 contour found";

                // just to avoid future bugs
                imwrite(QString(outputPath + "dataset/" + QString::number(i * countPerNumber + j) + ".png").toStdString(), mat);
            }
            else {
                RotatedRect rect = minAreaRect(contours[0]);

                float angle = rect.angle;

                Size rectSize = rect.size;
                if (rect.angle <= -35) {
                    angle += 90.0;
                    swap(rectSize.width, rectSize.height);
                }
                Mat rotationMatrix = getRotationMatrix2D(rect.center, angle, 1.0);
                Mat rotated, cropped;
                warpAffine(mat, rotated, rotationMatrix, mat.size(), INTER_CUBIC);
                getRectSubPix(rotated, rectSize, rect.center, cropped);

                Mat resized;
                resize(cropped, resized, Size(width, height));

                imwrite(QString(outputPath + "dataset/" + QString::number(i * countPerNumber + j) + ".png").toStdString(), resized);
            }

            labels += " " + QString::number(numbers[i]);
        }
    }

    QFile file(outputPath + "labels.txt");
    if (file.open(QIODevice::WriteOnly)){
        QTextStream stream(&file);
        stream << labels << endl;
        file.close();
    }
    else {
        qDebug() << "Cannot open " + outputPath + "labels.txt";
    }
}

/**
 * @brief generateSVM Generates an SVM.
 * @param path The path of the svm folder.
 * @param num The number of the svm.
 */
void generateSVM(QString path, int num1, int num2, int mode){
    int count = DATASET_COUNT;
    int totalCount = 10 * count;

    if (mode == 1){
        totalCount = 2 * count;
    }

    int dim = VECTOR_DIMENSION;

    float labels[totalCount];
    float* trainingData = new float[totalCount * dim];

    for (int i = 0; i < totalCount; i++){
        cv::Mat image;

        if (mode == 0){
            if (i / 10 == num1){
                image = cv::imread(("dataset/" + QString::number(num1) + "/" + QString::number(i%10) + ".png").toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
                labels[i] = num1;
            }
            else {
                image = cv::imread(("dataset/" + QString::number(i/10) + "/" + QString::number(i%10) + ".png").toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
                labels[i] = num1+1;
            }
        }
        else {
            if (i < count){
                image = cv::imread(("dataset/" + QString::number(num1) + "/" + QString::number(i) + ".png").toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
                labels[i] = num1;
            }
            else {
                image = cv::imread(("dataset/" + QString::number(num2) + "/" + QString::number(i-count) + ".png").toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
                labels[i] = num2;
            }
        }

        Skeleton ske(image);

        QList<double> vect = ske.vectorization();

        for (int j = 0; j < dim; j++){
            trainingData[i * dim + j] = vect[j];
        }
    }
    Mat labelsMat(totalCount, 1, CV_32SC1, labels);
    Mat trainingDataMat(totalCount, dim, CV_32FC1, trainingData);

    Ptr<ml::SVM> svm = ml::SVM::create();

    Ptr<TrainData> data = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);

    svm->setKernel(cv::ml::SVM::RBF);
    svm->setGamma(100);
    svm->setC(1000);
    svm->trainAuto(data);
    svm->save((path + "svm.xml").toStdString());

    delete trainingData;
}
