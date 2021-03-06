#ifndef SKELETON_H
#define SKELETON_H

#include <QSet>
#include <QList>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "headers.h"

#define MERGE_DISTANCE 0.05
#define DELETE_DISTANCE 0.15
#define FAKE_LOOPS_DISTANCE 0.15
#define JUNCTION_MARGIN 0.05

#define PART_X 7
#define PART_Y 10

#define SMALL_SIZE_X 9
#define SMALL_SIZE_Y 12

#define END_COUNT 0

#define JUNCTION_COUNT 0

#define HOLE_COUNT 0

#define MASS_CENTER_COUNT 0

#define TOTAL_COUNT 0

#define PARTS_COUNT 0

#define SMALL_SIZE_COUNT 1

#define VECTOR_DIMENSION (2*END_COUNT + 2*JUNCTION_COUNT + 2*HOLE_COUNT + 2*MASS_CENTER_COUNT + TOTAL_COUNT + PART_X*PART_Y*PARTS_COUNT + SMALL_SIZE_X*SMALL_SIZE_Y*SMALL_SIZE_COUNT)

struct LabeledPoint {
    int label;
    cv::Point2i point;
};

/**
 * @brief The Skeleton class represents a skeletonized version of a digit.
 */
class Skeleton {

    public:
        Skeleton(cv::Mat normalImage);

        int mostProbableDigit();
        QList<double> vectorization();

        cv::Point2d getMassCenter(cv::Mat ske);
        double getCount(cv::Mat ske);
        void setParts(cv::Mat ske);
        void setSmallImage(cv::Mat image);

        static double min(double a, double b);
        static double max(double a, double b);
        static int min(int a, int b);
        static int max(int a, int b);
        static QList<cv::Point2d> sort(QList<cv::Point2d> list);

    private:
        QList<LabeledPoint> startList;
        QList<cv::Point2d> listHoles;
        QList<cv::Point2d> listFakeHoles;
        QList<cv::Point2d> listJunctions;
        QList<cv::Point2d> listLineEnds;
        cv::Point2d massCenter;
        double total;
        double parts[PART_X][PART_Y];
        double smallImage[SMALL_SIZE_X][SMALL_SIZE_Y];
};

#endif // SKELETON_H
