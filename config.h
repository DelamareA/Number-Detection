#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <QString>
#include <QFile>
#include <QTextStream>
#include <QSet>
#include <opencv2/opencv.hpp>

#define CONFIG_LINES_COUNT 11

typedef cv::Ptr<cv::ml::SVM>* SVMs;

/**
 * @brief The Config class is used to contain all global information for the program.
 */
class Config {
    public:

        static void setConfigFromFile(QString path);
        static bool getIsVideo();
        static bool getIsMogUsed();
        static bool getIsPostProcessActivated();
        static QString getImagePath();
        static QString getVideoPath();
        static QString getBackgroundPath();
        static int getMaxBackgroundColorDistance();
        static QString getOutputVideoPath();
        static QString getOutputTextPath();
        static QSet<int> getDigitsOnField();
        static QSet<int> getNumbersOnField();
        static SVMs getSVMs();
        static SVMs getPairSVMs();

    private:
        Config();
        static bool isVideo;
        static bool isMogUsed;
        static bool isPostProcessActivated;
        static QString imagePath;
        static QString videoPath;
        static QString backgroundPath;
        static int maxBackgroundColorDistance;
        static QString outputVideoPath;
        static QString outputTextPath;
        static QSet<int> digitsOnField;
        static QSet<int> numbersOnField;
        static SVMs machines;
        static SVMs machinesPair;
};

#endif // CONFIGURATION_H
