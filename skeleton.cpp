#include <QDebug>
#include "skeleton.h"

/**
 * @brief Skeleton::Skeleton Main constructor for a skeleton.
 * @param normalImage The image of the digit, in black and white.
 */
Skeleton::Skeleton(cv::Mat normalImage){
    cv::Mat skeletonizedImage = thinningGuoHall(normalImage);
    QVector<cv::Point2i> dummyJunctionList;
    QVector<LabeledPoint> removedPoints;
    QVector<int> survivors;
    int currentLabel = 1;

    // SEARCH FOR ALL CURRENT BRANCHES

    for (int x = 0; x < skeletonizedImage.cols; x++){
        for (int y = 0; y < skeletonizedImage.rows; y++){
            if (skeletonizedImage.at<uchar>(y, x) == 255){
                int count = 0;
                QVector<cv::Point2i> list;
                for (int i = -1; i <= 1; i++){
                    for (int j = -1; j <= 1; j++){
                        if (i != 0 || j != 0){
                            if (y+j >= 0 && y+j < skeletonizedImage.rows && x+i >= 0 && x+i < skeletonizedImage.cols && skeletonizedImage.at<uchar>(y+j, x+i) > 0){
                                bool neighbourg = false;
                                cv::Point2i point(x+i,y+j);

                                for (int k = 0; k < list.size(); k++){
                                    if (cv::norm(point - list[k]) == 1){
                                        neighbourg = true;
                                    }
                                }

                                if (!neighbourg){
                                    count++;
                                    list.append(point);
                                }
                            }
                        }
                    }
                }

                if (count == 1){
                    cv::Point2i point(x,y);
                    LabeledPoint lp;
                    lp.label = currentLabel;
                    lp.point = point;
                    startList.append(lp);

                    currentLabel++;
                }
                else if (count > 2){
                    cv::Point2i point(x,y);
                    dummyJunctionList.append(point);
                }
            }
        }
    }

    // GO THROUGH THE BRANCH AND REMOVE ONE PIXEL AT A TIME

    for (int i = 0; i < startList.size(); i++){
        cv::Point2i current = startList[i].point;
        int label = startList[i].label;

        int round = 0;
        bool done = false;

        do {
            LabeledPoint lp;
            lp.point = current;
            lp.label = label;
            removedPoints.append(lp);

            skeletonizedImage.at<uchar>(current.y, current.x) = 0;

            int count = 0;
            int x = current.x;
            int y = current.y;
            QVector<cv::Point2i> list;
            for (int i = -1; i <= 1; i++){
                for (int j = -1; j <= 1; j++){
                    if (i != 0 || j != 0){
                        if (y+j >= 0 && y+j < skeletonizedImage.rows && x+i >= 0 && x+i < skeletonizedImage.cols && skeletonizedImage.at<uchar>(y+j, x+i) == 255){
                            bool junction = false;
                            cv::Point2i point(x+i,y+j);

                            for (int k = 0; k < dummyJunctionList.size(); k++){
                                if (cv::norm(point - dummyJunctionList[k]) <= 1){
                                    junction = true;
                                }
                            }

                            if (!junction){
                                count++;
                                list.append(point);
                            }
                        }
                    }
                }
            }

            if (count >= 1){
                current = list[0];
            }

            round ++;
            done = (count < 1);

            if (round == 12){
                survivors.append(label);
            }

        } while(round < 12 && !done);
    }

    // IF THE BRANCH IS STILL LONG ENOUGH, REDRAW IT

    for (int i = 0; i < removedPoints.size(); i++){
        cv::Point2i point = removedPoints[i].point;
        int label = removedPoints[i].label;

        if (survivors.contains(label)){
            skeletonizedImage.at<uchar>(point) = 255;
        }
    }

    // LOOPS

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::Mat imageClone = normalImage.clone();
    cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));
    cv::dilate(imageClone, imageClone, dilateElement);

    findContours(imageClone.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    cv::Mat invertedImage = normalImage.clone();

    if (contours.size() < 1){
        qDebug() << "Error, 0 connected components detected";
    }
    else {
        for (int x = 0; x < normalImage.cols; x++){
            for (int y = 0; y < normalImage.rows; y++){
                if (imageClone.at<uchar>(y, x) == 255){
                    invertedImage.at<uchar>(y, x) = 0;
                }
                else if (pointPolygonTest(contours[0], cv::Point2f(x,y), true) >= 0){
                    invertedImage.at<uchar>(y, x) = 255;
                }
                else {
                    invertedImage.at<uchar>(y, x) = 0;
                }
            }
        }
    }

    std::vector<std::vector<cv::Point> > invertedContours;
    std::vector<cv::Vec4i> invertedHierarchy;

    findContours(invertedImage, invertedContours, invertedHierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    for (unsigned int i = 0; i < invertedContours.size(); i++){
        cv::Rect rect = minAreaRect(invertedContours[i]).boundingRect();

        double xNorm = ((double) rect.x + rect.width/2) / skeletonizedImage.cols;
        double yNorm = ((double) rect.y + rect.height/2) / skeletonizedImage.rows;

        cv::Point2d point(xNorm, yNorm);


        if (rect.width > 2 && rect.height > 2){
            listHoles.push_back(point);
        }
        else {
            listFakeHoles.push_back(point);
        }
    }

    // COUNTING JUNCTIONS AND LINE ENDS


    for (int x = 0; x < skeletonizedImage.cols; x++){
        for (int y = 0; y < skeletonizedImage.rows; y++){
            if (skeletonizedImage.at<uchar>(y, x) == 255){
                int count = 0;
                QVector<cv::Point2i> list;
                for (int i = -1; i <= 1; i++){
                    for (int j = -1; j <= 1; j++){
                        if (i != 0 || j != 0){
                            if (y+j >= 0 && y+j < skeletonizedImage.rows && x+i >= 0 && x+i < skeletonizedImage.cols && skeletonizedImage.at<uchar>(y+j, x+i) == 255){
                                bool neighbourg = false;
                                cv::Point2i point(x+i,y+j);

                                for (int k = 0; k < list.size(); k++){
                                    if (cv::norm(point - list[k]) == 1){
                                        neighbourg = true;
                                    }
                                }

                                if (!neighbourg){
                                    count++;
                                    list.append(point);
                                }
                            }
                        }
                    }
                }

                if (count == 1 || count > 2){
                    double xNorm = ((double) x) / skeletonizedImage.cols;
                    double yNorm = ((double) y) / skeletonizedImage.rows;
                    cv::Point2d point(xNorm,yNorm);

                    if (count == 1){
                        listLineEnds.push_back(point);
                    }
                    else {
                        listJunctions.push_back(point);
                    }
                }


            }
        }
    }

    // MERGING CLOSE JUNCTIONS
    bool done = true;

    do {
        done = true;
        int keepIndexJunction = -1;
        int removeIndexJunction = -1;
        for (int i = 0; i < listJunctions.size(); i++){
            for (int j = 0; j < listJunctions.size(); j++){
                if (i != j && norm(listJunctions[i] - listJunctions[j]) < MERGE_DISTANCE){
                    keepIndexJunction = i;
                    removeIndexJunction = j;
                }
            }
        }

        if (keepIndexJunction != -1 && removeIndexJunction != -1){
            done = false;
            listJunctions[keepIndexJunction].x = (listJunctions[keepIndexJunction].x + listJunctions[removeIndexJunction].x) / 2;
            listJunctions[keepIndexJunction].y = (listJunctions[keepIndexJunction].y + listJunctions[removeIndexJunction].y) / 2;
            listJunctions.removeAt(removeIndexJunction);
        }
    } while (!done);

    // DELETING CLOSE JUNCTIONS AND LINE ENDS, ALWAYS WRONG DATA
    done = true;

    do {
        done = true;
        int removeIndexLineEnds = -1;
        int removeIndexJunctions = -1;
        for (int i = 0; i < listLineEnds.size(); i++){
            for (int j = 0; j < listJunctions.size(); j++){
                if (norm(listLineEnds[i] - listJunctions[j]) < DELETE_DISTANCE){
                    removeIndexLineEnds = i;
                    removeIndexJunctions = j;
                }
            }
        }

        if (removeIndexLineEnds != -1 && removeIndexJunctions != -1){
            done = false;
            listLineEnds.removeAt(removeIndexLineEnds);
            listJunctions.removeAt(removeIndexJunctions);
        }
    } while (!done);

    // DELETING JUNCTIONS CLOSE TO FAKE LOOPS

    done = true;

    do {
        done = true;
        int removeIndexJunction = -1;
        for (int i = 0; i < listJunctions.size(); i++){
            for (int j = 0; j < listFakeHoles.size(); j++){
                if (norm(listJunctions[i] - listFakeHoles[j]) < FAKE_LOOPS_DISTANCE){
                    removeIndexJunction = i;
                }
            }
        }

        if (removeIndexJunction != -1){
            done = false;
            listJunctions.removeAt(removeIndexJunction);
        }
    } while (!done);

    // DELETING LINE ENDS CLOSE TO FAKE LOOPS

    done = true;

    do {
        done = true;
        int removeIndexEnd = -1;
        for (int i = 0; i < listLineEnds.size(); i++){
            for (int j = 0; j < listFakeHoles.size(); j++){
                if (norm(listLineEnds[i] - listFakeHoles[j]) < FAKE_LOOPS_DISTANCE){
                    removeIndexEnd = i;
                }
            }
        }

        if (removeIndexEnd != -1){
            done = false;
            listLineEnds.removeAt(removeIndexEnd);
        }
    } while (!done);

    // DELETING JUNCTIONS CLOSE TO BORDERS, ALWAYS WRONG DATA
    done = true;

    do {
        done = true;
        int removeIndexJunctions = -1;
        for (int i = 0; i < listJunctions.size(); i++){
            if (listJunctions[i].x < JUNCTION_MARGIN || listJunctions[i].y < JUNCTION_MARGIN || listJunctions[i].x > 1 - JUNCTION_MARGIN || listJunctions[i].y > 1 - JUNCTION_MARGIN){
                removeIndexJunctions = i;
            }
        }

        if (removeIndexJunctions != -1){
            done = false;
            listJunctions.removeAt(removeIndexJunctions);
        }
    } while (!done);

    massCenter = getMassCenter(skeletonizedImage);

    total = getCount(skeletonizedImage);

    setParts(skeletonizedImage);
    setSmallImage(normalImage);

    listLineEnds = sort(listLineEnds);
    listHoles = sort(listHoles);
    listJunctions = sort(listJunctions);
}

/**
 * @brief Skeleton::mostProbableDigit Outputs the most probable digit of the skeleton.
 * @return the most probable digit of the skeleton.
 */
int Skeleton::mostProbableDigit(){
    QSet<int> digitsOnField = Config::getDigitsOnField();

    int dim = VECTOR_DIMENSION;

    QList<double> vect = vectorization();

    float sampleData[dim];

    for (int i = 0; i < dim; i++){
        sampleData[i] = vect[i];
    }

    cv::Mat sampleMat(1, dim, CV_32FC1, sampleData);

    // One against all

    QList<int> electedDigits;

    for (int i = 0; i < 10; i++){
        if (digitsOnField.contains(i)) {
            float response = Config::getSVMs()[i]->predict(sampleMat);

            if (response == 0.0){
                electedDigits.push_back(i);
            }
        }
    }

    if (electedDigits.size() == 1){
        //qDebug() << "Only elected" << electedDigits[0];
        return electedDigits[0];
    }
    else {
        //qDebug() << "Elected" << electedDigits.size();
    }

    // One against one

    int maxVote = 0;
    int intMaxVote = -1;

    for (int i = 0; i < 10; i++){
        int vote = 0;
        for (int j = 0; j < 10; j++){
            if (i != j && digitsOnField.contains(i) && digitsOnField.contains(j)) {
                float response = Config::getPairSVMs()[min(i,j) * 10 + max(i,j)]->predict(sampleMat);

                if ((response == 0.0 && i == min(i,j)) || (response == 1.0 && i == max(i,j))){
                    vote++;
                }
            }
        }

        //qDebug() << i << ": " << vote;

        if (vote > maxVote){
            maxVote = vote;
            intMaxVote = i;
        }
    }

    if (intMaxVote == -1){
        qDebug() << "Error : No vote is greater than 0";
        return -1;
    }
    else {
        return intMaxVote;
    }
}

/**
 * @brief Skeleton::vectorization Outputs a vectorized representation of the skeleton.
 * @return a vectorized representation of the skeleton, which depends on the values of END_COUNT, HOLE_COUNT, etc.
 */
QList<double> Skeleton::vectorization() {
    QList<double> result;

    int index = 0;

    for (int i = 0; i < VECTOR_DIMENSION; i++){
        result.push_back(0.0001);
    }

    for (int i = 0; i < listLineEnds.size() && i < END_COUNT; i++){
        result[index] = (listLineEnds[i].x);
        index++;

        result[index] = (listLineEnds[i].y);
        index++;
    }

    for (int i = 0; i < listJunctions.size() && i < JUNCTION_COUNT; i++){
        result[index] = (listJunctions[i].x);
        index++;

        result[index] = (listJunctions[i].y);
        index++;
    }

    for (int i = 0; i < listHoles.size() && i < HOLE_COUNT; i++){
        result[index] = (listHoles[i].x);
        index++;

        result[index] = (listHoles[i].y);
        index++;
    }

    for (int i = 0; i < MASS_CENTER_COUNT; i++){
        result[index] = (massCenter.x);
        index++;

        result[index] = (massCenter.y);
        index++;
    }

    for (int i = 0; i < TOTAL_COUNT; i++){
        result[index] = (total);
        index++;
    }

    for (int i = 0; i < PARTS_COUNT; i++){
        for (int x = 0; x < PART_X; x++){
            for (int y = 0; y < PART_Y; y++){
                result[index] = parts[x][y];
                index++;
            }
        }
    }

    for (int i = 0; i < SMALL_SIZE_COUNT; i++){
        for (int x = 0; x < SMALL_SIZE_X; x++){
            for (int y = 0; y < SMALL_SIZE_Y; y++){
                result[index] = smallImage[x][y];
                index++;
            }
        }
    }

    return result;
}

/**
 * @brief Skeleton::getMassCenter Outputs the mass center of the skeleton.
 * @param ske The image of the skeleton.
 * @return the mass center of the skeleton, x and y are between 0 and 1.
 */
cv::Point2d Skeleton::getMassCenter(cv::Mat ske){
    int sumX = 0;
    int sumY = 0;
    double count = 0;
    for (int i = 0; i < ske.cols; i++){
        for (int j = 0; j < ske.rows; j++){
            if (ske.at<uchar>(j,i) == 255){
                sumX += i;
                sumY += j;
                count ++;
            }
        }
    }

    double tempX = (sumX / count) / ske.cols;
    double tempY = (sumY / count) / ske.rows;

    return cv::Point2d(tempX, tempY);
}

/**
 * @brief Skeleton::getCount Outputs the total number of white pixels in the skeleton.
 * @param ske The skeleton image.
 * @return A value between 0 and 1 representing the total number of white pixels in the skeleton.
 */
double Skeleton::getCount(cv::Mat ske){
    double count = 0;
    for (int i = 0; i < ske.cols; i++){
        for (int j = 0; j < ske.rows; j++){
            if (ske.at<uchar>(j,i) == 255){
                count ++;
            }
        }
    }

    double tempCount = count / (ske.cols * ske.rows);

    // increase value, for better detection results, otherwise, each number is very close to 0
    return min(tempCount * 20, 1.0);
}

/**
 * @brief Skeleton::setParts Set the number of white pixels for each part of the image.
 * @param ske The skeleton image.
 */
void Skeleton::setParts(cv::Mat ske){
    for (int x = 0; x < PART_X; x++){
        for (int y = 0; y < PART_Y; y++){
            parts[x][y] = 0;
        }
    }

    for (int i = 0; i < ske.cols; i++){
        for (int j = 0; j < ske.rows; j++){
            if (ske.at<uchar>(j,i) == 255){
                parts[(PART_X * i) / ske.cols][(PART_Y * j) / ske.rows] ++;
            }
        }
    }

    // normalization
    for (int x = 0; x < PART_X; x++){
        for (int y = 0; y < PART_Y; y++){
            parts[x][y] = (parts[x][y] / ((ske.cols / PART_X) * (ske.rows / PART_Y)));
        }
    }

    // increase value, for better detection results, otherwise, each number is very close to 0
    for (int x = 0; x < PART_X; x++){
        for (int y = 0; y < PART_Y; y++){
            parts[x][y] = min(parts[x][y] * 20, 1.0);
        }
    }
}

void Skeleton::setSmallImage(cv::Mat image){
    cv::Mat small;
    cv::resize(image, small, cv::Size(SMALL_SIZE_X, SMALL_SIZE_Y));
    for (int x = 0; x < SMALL_SIZE_X; x++){
        for (int y = 0; y < SMALL_SIZE_Y; y++){
            smallImage[x][y] = small.at<uchar>(y,x) / 255.0;
        }
    }
}

double Skeleton::min(double a, double b){
    return a < b ? a : b;
}

double Skeleton::max(double a, double b){
    return a > b ? a : b;
}

int Skeleton::min(int a, int b){
    return a < b ? a : b;
}

int Skeleton::max(int a, int b){
    return a > b ? a : b;
}

QList<cv::Point2d> Skeleton::sort(QList<cv::Point2d> list){
    QList<cv::Point2d> result;

    for (int i = 0; i < list.count(); i++){
        double min = 1.1;
        int index = -1;
        for (int j = 0; j < list.count(); j++){
            if (list[j].y < min){
                min = list[j].y;
                index = j;
            }

        }

        if (index != -1){
            cv::Point2d point(list[index].x, list[index].y);
            result.push_back(point);
            list[index].y = 1.1;
        }
        else {
            qDebug() << "Error in sort, some value must be bigger than 1.0";
        }
    }

    return result;
}
