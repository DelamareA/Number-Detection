#ifndef FUNCTIONS
#define FUNCTIONS

#include "output.h"
#include "template.h"
#include "skeleton.h"
#include "configuration.h"
#include <opencv2/core.hpp>

#define MODULES_COUNT 5
#define FUNCTIONS_COUNT 1
#define ROTATION_STEP 10
#define ROTATION_MAX 30
#define ROTATION_COUNT (2*ROTATION_MAX/ROTATION_STEP + 1)

enum {TEMPLATE_MATCHING, CENTER_MASS, HALVES_CENTER_MASS_VERTI, HALVES_CENTER_MASS_HORI, HISTOGRAMS};

struct NumPos {
    int number;
    cv::Point2i pos;
};

Output* templateMatching(cv::Mat image, int modules[MODULES_COUNT], cv::Mat backgroundLab, QList<int> digitsOnField);

Output* basicTemplateMatching(cv::Mat image, cv::Mat background, QList<int> digitsOnField);

int colorDistance(cv::Vec3b c1, cv::Vec3b c2);

cv::Mat extractBackgroundFromVideo(QString fileName, int maxFrames);
cv::Mat extractBackgroundFromVideo2(QString fileName, int maxFrames);
cv::Mat extractBackgroundFromFiles(QStringList filesName);

cv::Mat closeGaps(cv::Mat image, int patchSize, double ratio);

cv::Point2f getMassCenterFromImage(cv::Mat image);
cv::Mat getSkeleton(cv::Mat image);

cv::Mat thinningGuoHall(cv::Mat image);
void thinningGuoHallIteration(cv::Mat& im, int iter);

void generateDataSet(QList<int> numbers, int countPerNumber, int width, int height, QString outputPath);
void generateSVM(QString path, int type);
void generateJerseySVM(QString path);

NumPos mostProbableNumber(cv::Mat image, QList<int> digitsOnField);
int mostProbableDigit(cv::Mat numberImage, QList<int> digitsOnField, std::vector<cv::Point> contour);
int digitHelper(cv::Mat bigImage, QList<int> digitsOnField, std::vector<cv::Point> contour, cv::Rect rect);
void runOnDataSet(QList<int> digitsOnField);

#endif // FUNCTIONS

