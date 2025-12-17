#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

int main() {
    std::vector<std::string> msg = {"Hello", "C++", "World", "from", "VSCode", "Template!"};

    for (const auto& word : msg) {
        std::cout << word << " ";
    }
    std::cout << std::endl;

    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    return 0;
}
