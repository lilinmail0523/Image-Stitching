

cv::Mat cylindricalProjection(cv::Mat img, double f, std::vector<Point>& pts);

std::pair<int, int> calculateShift(std::vector<Point>& ptsLeft, std::vector<Point>& ptsRight, int prev_width, double threshold = 25.0, int groupsize = 4);

void alphaBlending(cv::Mat& Img, cv::Mat ImagetoBlend, int move_x, int move_y, bool DirectConnection = false);


std::vector<cv::Mat> gaussianPyramid(cv::Mat img, int leveln);

std::vector<cv::Mat> LaplacianPyramid(std::vector<cv::Mat>& GaussianImgs, int leveln);

void multibandBlending(cv::Mat& Img, cv::Mat ImagetoBlend, int move_x, int move_y, int leveln = 4);


