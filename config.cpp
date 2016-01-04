#include <QDebug>
#include "config.h"

bool Config::isVideo = false;
bool Config::isMogUsed = false;
bool Config::isPostProcessActivated = false;
QString Config::imagePath = "";
QString Config::videoPath = "";
QString Config::backgroundPath = "";
int Config::maxBackgroundColorDistance = 30;
QString Config::outputVideoPath = "";
QString Config::outputTextPath = "";
QSet<int> Config::digitsOnField;
QSet<int> Config::numbersOnField;
SVMs Config::machines;
SVMs Config::machinesPair;

Config::Config(){
}

/**
 * @brief Config::setConfigFromFile Initializes the configuration for the programm.
 * @param path The path of the config file.
 */
void Config::setConfigFromFile(QString path){
    QFile file(path);
    if (file.open(QIODevice::ReadOnly)){
        QString lines;
        QTextStream stream(&file);
        lines = stream.readAll();
        file.close();

        QStringList list = lines.split('\n');

        if (list.size() == CONFIG_LINES_COUNT){
        for (int i = 0; i < list.size(); i++){
            QStringList l = list[i].replace("\r", "").split(':');

            if (l[0] == "isVideo"){
                Config::isVideo = l[1].toInt();
            }
            else if (l[0] == "isMogUsed"){
                Config::isMogUsed = l[1].toInt();
            }
            else if (l[0] == "isPostProcessActivated"){
                Config::isPostProcessActivated = l[1].toInt();
            }
            else if (l[0] == "imagePath"){
                Config::imagePath = l[1];
            }
            else if (l[0] == "videoPath"){
                Config::videoPath = l[1];
            }
            else if (l[0] == "backgroundPath"){
                Config::backgroundPath = l[1];
            }
            else if (l[0] == "maxBackgroundColorDistance"){
                Config::maxBackgroundColorDistance = l[1].toInt();
            }
            else if (l[0] == "outputVideoPath"){
                Config::outputVideoPath = l[1];
            }
            else if (l[0] == "outputTextPath"){
                Config::outputTextPath = l[1];
            }
            else if (l[0] == "numbersOnField"){
                numbersOnField.clear();
                digitsOnField.clear();

                for (int j = 1; j < l.size(); j++){
                    int num = l[j].toInt();

                    if (num < 1 || num > 99){
                        qDebug() << "Invalid number" << num;
                    }
                    else {
                        numbersOnField.insert(num);

                        if (num >= 10){
                            digitsOnField.insert(num/10);
                            digitsOnField.insert(num%10);
                        }
                        else {
                            digitsOnField.insert(num);
                        }
                    }
                }
            }
            else if (l[0] == "svmsPath"){
                Config::machines = new cv::Ptr<cv::ml::SVM>[10];
                Config::machinesPair = new cv::Ptr<cv::ml::SVM>[10 * 10];

                for (int i = 0; i < 10; i++){
                    Config::machines[i] = cv::ml::SVM::load<cv::ml::SVM>(QString(l[1] + "/" + QString::number(i) + "-all/svm.xml").toStdString());
                    for (int j = i+1; j < 10; j++){
                        Config::machinesPair[i * 10 + j] = cv::ml::SVM::load<cv::ml::SVM>(QString(l[1] + "/" + QString::number(i) + "-" + QString::number(j) + "/svm.xml").toStdString());
                    }
                }
            }
            else {
                qDebug() << "Unknown config line" << l[0];
            }
        }
        }
        else {
            qDebug() << "Config file incorrect";
            exit(-1);
        }
    }
    else {
        qDebug() << "Cannot open " + path;
        exit(-1);
    }
}

bool Config::getIsVideo(){
    return Config::isVideo;
}

bool Config::getIsMogUsed(){
    return Config::isMogUsed;
}

bool Config::getIsPostProcessActivated(){
    return Config::isPostProcessActivated;
}

QString Config::getImagePath(){
    return Config::imagePath;
}

QString Config::getVideoPath(){
    return Config::videoPath;
}

QString Config::getBackgroundPath(){
    return Config::backgroundPath;
}

int Config::getMaxBackgroundColorDistance(){
    return Config::maxBackgroundColorDistance;
}

QString Config::getOutputVideoPath(){
    return Config::outputVideoPath;
}

QString Config::getOutputTextPath(){
    return Config::outputTextPath;
}

QSet<int> Config::getDigitsOnField(){
    return Config::digitsOnField;
}

QSet<int> Config::getNumbersOnField(){
    return Config::numbersOnField;
}

SVMs Config::getSVMs(){
    return Config::machines;
}

SVMs Config::getPairSVMs(){
    return Config::machinesPair;
}
