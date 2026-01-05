#pragma once

#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <opencv2/opencv.hpp>

#include "../style_transfer/AdaIN.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void onSelectContent();
    void onSelectStyle();
    void onApply();

private:
    void setupUi();
    void displayImage(QLabel* label, const cv::Mat& img);

    // UI Elements
    QWidget* centralWidget;
    QLabel* contentLabel;
    QLabel* styleLabel;
    QLabel* resultLabel;
    QPushButton* btnSelectContent;
    QPushButton* btnSelectStyle;
    QPushButton* btnApply;

    // Data
    cv::Mat contentImage;
    cv::Mat styleImage;
    cv::Mat resultImage;

    // Style Transfer logic
    AdaIN* adain;  // We'll create this on demand or keep it
};
