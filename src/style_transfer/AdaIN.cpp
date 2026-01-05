#include "AdaIN.h"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <vector>

// Initialize static members
cv::dnn::Net AdaIN::encoderNet;
cv::dnn::Net AdaIN::decoderNet;
bool AdaIN::modelsLoaded = false;

void AdaIN::loadModels() {
    if (modelsLoaded)
        return;

    // Standard paths relative to executable location or project structure
    // Standard paths relative to executable location or project structure
    std::vector<std::string> searchDirs = {"models/", "../models/", "../../models/", "AI_models/",
                                           "../Resources/AI_models/"};

    std::string encPath, decPath;
    bool encFound = false;
    bool decFound = false;

    // Look for converted ONNX models
    std::string encoderName = "vgg_normalised.onnx";
    std::string decoderName = "decoder.onnx";

    for (const auto& dir : searchDirs) {
        std::string testEnc = dir + encoderName;
        std::string testDec = dir + decoderName;

        FILE* fEnc = fopen(testEnc.c_str(), "r");
        if (fEnc) {
            fclose(fEnc);
            encPath = testEnc;
            encFound = true;
        }

        FILE* fDec = fopen(testDec.c_str(), "r");
        if (fDec) {
            fclose(fDec);
            decPath = testDec;
            decFound = true;
        }

        if (encFound && decFound)
            break;
    }

    if (!encFound || !decFound) {
        std::cerr << "❌ AdaIN Error: Could not find VGG encoder/decoder models." << std::endl;
        return;
    }

    try {
        encoderNet = cv::dnn::readNet(encPath);
        decoderNet = cv::dnn::readNet(decPath);

        if (encoderNet.empty() || decoderNet.empty()) {
            std::cerr << "Failed to load AdaIN models." << std::endl;
            return;
        }

        encoderNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        encoderNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        decoderNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        decoderNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        modelsLoaded = true;
        std::cout << "✅ AdaIN models loaded successfully." << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading AdaIN models: " << e.what() << std::endl;
        modelsLoaded = false;
    }
}

AdaIN::AdaIN(const cv::Mat& styleImage, float alpha) : m_styleImage(styleImage), m_alpha(alpha) {
    loadModels();
}

AdaIN::AdaIN(const std::string& stylePath, float alpha) : m_alpha(alpha) {
    m_styleImage = cv::imread(stylePath);
    if (m_styleImage.empty()) {
        std::cerr << "⚠️ AdaIN Warning: Could not load style image from " << stylePath << std::endl;
    }
    loadModels();
}

AdaIN::~AdaIN() {}

void AdaIN::calcMeanStd(const cv::Mat& features, cv::Mat& mean, cv::Mat& std) {
    if (features.dims != 4)
        return;

    int C = features.size[1];
    int H = features.size[2];
    int W = features.size[3];

    mean = cv::Mat::zeros(C, 1, CV_32F);
    std = cv::Mat::zeros(C, 1, CV_32F);

    for (int c = 0; c < C; ++c) {
        cv::Mat channel(H, W, CV_32F, (void*)features.ptr<float>(0, c));
        cv::Scalar m, s;
        cv::meanStdDev(channel, m, s);
        mean.at<float>(c) = (float)m[0];
        std.at<float>(c) = (float)s[0];
    }
}

cv::Mat AdaIN::adain(const cv::Mat& contentFeats, const cv::Mat& styleFeats) {
    cv::Mat contentMean, contentStd;
    cv::Mat styleMean, styleStd;

    calcMeanStd(contentFeats, contentMean, contentStd);
    calcMeanStd(styleFeats, styleMean, styleStd);

    cv::Mat targetFeats = contentFeats.clone();

    int C = contentFeats.size[1];
    int H = contentFeats.size[2];
    int W = contentFeats.size[3];

    const float eps = 1e-5f;

    for (int c = 0; c < C; ++c) {
        float cm = contentMean.at<float>(c);
        float cs = contentStd.at<float>(c);
        float sm = styleMean.at<float>(c);
        float ss = styleStd.at<float>(c);

        cv::Mat channel(H, W, CV_32F, targetFeats.ptr<float>(0, c));
        channel = (channel - cm) / (cs + eps);
        channel = channel * ss + sm;
    }

    return targetFeats;
}

cv::Mat AdaIN::apply(const cv::Mat& srcImg) {
    if (srcImg.empty() || m_styleImage.empty() || !modelsLoaded) {
        return srcImg;
    }

    // Prepare inputs
    // Prepare inputs
    // Resize if too large to prevent integer overflow in DNN module
    // and to speed up style transfer.
    int maxDim = 1024;
    cv::Mat processingImg = srcImg;
    double scale = 1.0;

    if (processingImg.cols > maxDim || processingImg.rows > maxDim) {
        if (processingImg.cols > processingImg.rows) {
            scale = (double)maxDim / processingImg.cols;
        } else {
            scale = (double)maxDim / processingImg.rows;
        }
        cv::resize(processingImg, processingImg, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    cv::Mat contentBlob = cv::dnn::blobFromImage(
        processingImg, 1.0, cv::Size(), cv::Scalar(103.939, 116.779, 123.68), false, false);
    cv::Mat styleBlob = cv::dnn::blobFromImage(m_styleImage, 1.0, cv::Size(),
                                               cv::Scalar(103.939, 116.779, 123.68), false, false);

    // Encode Content
    encoderNet.setInput(contentBlob);
    cv::Mat contentFeats = encoderNet.forward();

    // Encode Style
    encoderNet.setInput(styleBlob);
    cv::Mat styleFeats = encoderNet.forward();

    // AdaIN
    cv::Mat t = adain(contentFeats, styleFeats);

    // Alpha blending
    if (m_alpha < 1.0f) {
        t = t * m_alpha + contentFeats * (1.0f - m_alpha);
    }

    // Decode
    decoderNet.setInput(t);
    cv::Mat outputBlob = decoderNet.forward();

    // Postprocess
    std::vector<cv::Mat> images;
    cv::dnn::imagesFromBlob(outputBlob, images);

    if (images.empty())
        return srcImg;

    cv::Mat result = images[0];
    result = result + cv::Scalar(103.939, 116.779, 123.68);

    // Clip and Convert
    cv::Mat finalImg;
    result.convertTo(finalImg, CV_8U);

    if (srcImg.channels() == 4) {
        if (finalImg.size() != srcImg.size()) {
            cv::resize(finalImg, finalImg, srcImg.size());
        }

        cv::Mat alphaCh;
        cv::extractChannel(srcImg, alphaCh, 3);
        cv::cvtColor(finalImg, finalImg, cv::COLOR_BGR2BGRA);
        std::vector<cv::Mat> chs;
        cv::split(finalImg, chs);
        chs[3] = alphaCh;
        cv::merge(chs, finalImg);
    } else {
        if (finalImg.size() != srcImg.size()) {
            cv::resize(finalImg, finalImg, srcImg.size());
        }
    }

    return finalImg;
}
