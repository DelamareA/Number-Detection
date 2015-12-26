#-------------------------------------------------
#
# Project created by QtCreator 2015-12-14T15:59:02
#
#-------------------------------------------------

INCLUDEPATH += C:\OpenCV\opencv-build\install\include
LIBS += -L"C:\OpenCV\opencv-build\install\x86\mingw\bin"
LIBS += -lopencv_core300 -lopencv_highgui300 -lopencv_imgproc300 -lopencv_calib3d300 -lopencv_features2d300 -lopencv_flann300 -lopencv_imgcodecs300 -lopencv_video300 -lopencv_videoio300 -lopencv_ml300

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Number-Detection
TEMPLATE = app


SOURCES += main.cpp \
    numberdetection.cpp \
    output.cpp \
    skeleton.cpp \
    svm.cpp \
    config.cpp \
    frameprocess.cpp

HEADERS  += \
    output.h \
    skeleton.h \
    config.h \
    headers.h

FORMS    +=
