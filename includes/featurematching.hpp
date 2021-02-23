
void plotMatching(std::vector<Point> Pts1, std::vector<Point> Pts2, std::vector<cv::Mat> warpedimgs);


std::tuple <std::vector<Point>, std::vector<Point>> KNNMatching(std::vector<Point>& Pts1, std::vector<Point>& Pts2);