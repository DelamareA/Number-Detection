#ifndef FUNCTIONS
#define FUNCTIONS

#include <opencv2/core.hpp>
#include "output.h"
#include "skeleton.h"
#include "config.h"

#define DATASET_COUNT 10

struct NumPos {
    int number;
    cv::Point2i pos;
};

enum FRAME_MODE {NORMAL, MOG};

Output* frameProcess(cv::Mat image, cv::Mat background, cv::Mat foregroundMask, FRAME_MODE mode);

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
void generateSVM(QString path, int num1, int num2);

NumPos mostProbableNumber(cv::Mat image);
int mostProbableDigit(cv::Mat digitImage, std::vector<cv::Point> contour);
int digitHelper(cv::Mat bigImage, std::vector<cv::Point> contour, cv::Rect rect);
void runOnDataSet();

#endif // FUNCTIONS

