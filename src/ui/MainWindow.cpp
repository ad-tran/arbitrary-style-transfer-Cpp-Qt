#include "MainWindow.h"

#include <QApplication>
#include <QFileDialog>
#include <QImage>
#include <QMessageBox>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent), adain(nullptr) { setupUi(); }

MainWindow::~MainWindow() {
    if (adain)
        delete adain;
}

void MainWindow::setupUi() {
    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    // Main Layout
    QVBoxLayout* mainLayout = new QVBoxLayout(centralWidget);

    // Top Section: Content and Style Images side-by-side
    QHBoxLayout* imageLayout = new QHBoxLayout();

    // Content Layout
    QVBoxLayout* contentLayout = new QVBoxLayout();
    contentLabel = new QLabel("Content Image");
    contentLabel->setAlignment(Qt::AlignCenter);
    contentLabel->setFrameShape(QFrame::Box);
    contentLabel->setMinimumSize(300, 300);
    btnSelectContent = new QPushButton("+ Content Image");
    contentLayout->addWidget(contentLabel);
    contentLayout->addWidget(btnSelectContent);

    // Style Layout
    QVBoxLayout* styleLayout = new QVBoxLayout();
    styleLabel = new QLabel("Style Image");
    styleLabel->setAlignment(Qt::AlignCenter);
    styleLabel->setFrameShape(QFrame::Box);
    styleLabel->setMinimumSize(300, 300);
    btnSelectStyle = new QPushButton("+ Style Image");
    styleLayout->addWidget(styleLabel);
    styleLayout->addWidget(btnSelectStyle);

    imageLayout->addLayout(contentLayout);
    imageLayout->addSpacing(20);
    imageLayout->addLayout(styleLayout);

    mainLayout->addLayout(imageLayout);

    // Middle Section: Apply Button
    btnApply = new QPushButton("Apply Style Transfer");
    btnApply->setFixedHeight(50);
    mainLayout->addWidget(btnApply);

    // Bottom Section: Result Image
    resultLabel = new QLabel("Result Image");
    resultLabel->setAlignment(Qt::AlignCenter);
    resultLabel->setFrameShape(QFrame::Box);
    resultLabel->setMinimumSize(400, 400);  // Make it slightly larger
    mainLayout->addWidget(resultLabel);

    // Connections
    connect(btnSelectContent, &QPushButton::clicked, this, &MainWindow::onSelectContent);
    connect(btnSelectStyle, &QPushButton::clicked, this, &MainWindow::onSelectStyle);
    connect(btnApply, &QPushButton::clicked, this, &MainWindow::onApply);

    setWindowTitle("Arbitrary Style Transfer App");
    resize(800, 800);
}

void MainWindow::onSelectContent() {
    QString fileName = QFileDialog::getOpenFileName(this, "Select Content Image", "",
                                                    "Images (*.png *.jpg *.jpeg *.bmp)");
    if (!fileName.isEmpty()) {
        contentImage = cv::imread(fileName.toStdString());
        if (!contentImage.empty()) {
            displayImage(contentLabel, contentImage);
        }
    }
}

void MainWindow::onSelectStyle() {
    QString fileName = QFileDialog::getOpenFileName(this, "Select Style Image", "",
                                                    "Images (*.png *.jpg *.jpeg *.bmp)");
    if (!fileName.isEmpty()) {
        styleImage = cv::imread(fileName.toStdString());
        if (!styleImage.empty()) {
            displayImage(styleLabel, styleImage);
        }
    }
}

void MainWindow::onApply() {
    if (contentImage.empty() || styleImage.empty()) {
        QMessageBox::warning(this, "Warning", "Please select both content and style images.");
        return;
    }

    resultLabel->setText("Processing...");
    QApplication::processEvents();  // Allow UI to update

    // Initialize AdaIN with selected style
    if (adain)
        delete adain;
    adain = new AdaIN(styleImage);

    // Apply
    resultImage = adain->apply(contentImage);

    if (!resultImage.empty()) {
        displayImage(resultLabel, resultImage);
    } else {
        resultLabel->setText("Failed to process image.");
    }
}

void MainWindow::displayImage(QLabel* label, const cv::Mat& img) {
    // Convert cv::Mat to QImage
    // OpenCV is BGR, Qt is RGB
    cv::Mat rgb;
    if (img.channels() == 3) {
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, rgb, cv::COLOR_BGRA2RGBA);
    } else {
        cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
    }

    QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);

    // Scale to fit label, keeping aspect ratio
    QPixmap pixmap = QPixmap::fromImage(qimg);
    label->setPixmap(pixmap.scaled(label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
