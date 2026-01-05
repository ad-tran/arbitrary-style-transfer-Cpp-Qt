#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>

class AdaIN {
   public:
    /**
     * @brief Constructor for AdaIN style transfer.
     * @param styleImage The style image to apply.
     * @param alpha Strength of style transfer (0.0 - 1.0).
     */
    AdaIN(const cv::Mat& styleImage, float alpha = 1.0f);

    /**
     * @brief Alternative constructor using path.
     * @param stylePath Path to the style image.
     * @param alpha Strength of style transfer (0.0 - 1.0).
     */
    AdaIN(const std::string& stylePath, float alpha = 1.0f);

    ~AdaIN();

    /**
     * @brief Applies the style transfer to the content image.
     * @param srcImg Content image.
     * @return Stylized image.
     */
    cv::Mat apply(const cv::Mat& srcImg);

   private:
    cv::Mat m_styleImage;
    float m_alpha;

    // Shared models (loaded once)
    static cv::dnn::Net encoderNet;
    static cv::dnn::Net decoderNet;
    static bool modelsLoaded;

    static void loadModels();

    /**
     * @brief Calculates channel-wise mean and standard deviation.
     */
    void calcMeanStd(const cv::Mat& features, cv::Mat& mean, cv::Mat& std);

    /**
     * @brief Performs Adaptive Instance Normalization.
     */
    cv::Mat adain(const cv::Mat& contentFeats, const cv::Mat& styleFeats);
};
