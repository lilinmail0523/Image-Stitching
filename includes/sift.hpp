

class SIFT {

public:

	std::string m_SIFT_filename;
	int m_SIFT_id;

	SIFT() { m_SIFT_id = -1; }
	SIFT(std::string filename, int id) : m_SIFT_filename(filename), m_SIFT_id(id) {}

	std::vector<Point> SIFTProcessing(cv::Mat Img);
	void plotPoint(cv::Mat img, std::vector<Point>& Pts);
private:
	std::vector<cv::Mat> m_ImgsPyramids;
	std::vector<std::vector<cv::Mat>> m_GaussianPyramids;
	std::vector<std::vector<cv::Mat>> m_DOGPyramids;
	std::vector<std::vector<double>> m_SigmaPyramids;


	void generateDOG();

	//oct = octave level, inl = interval #
	bool findExtrema(int oct, int x, int y, int inls);
	cv::Mat firstDerivation(int oct, double x, double y, int inl);
	cv::Mat secondDerovation(int oct, double x, double y, int inl);

	std::vector<std::vector<Point>>  findFeature();

	std::vector<Point> findOrientation(std::vector<std::vector<Point>> featurelist);

	std::vector<Point> createDescriptor(std::vector<Point>& features);
};


void test(std::vector<std::string> imglist);