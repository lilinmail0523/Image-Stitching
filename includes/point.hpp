#include<vector>


class Point {

public:


	int x;
	int y;
	int imageID;
	std::string imageName;
	int octavelevel;
	int interval;
	int maxOrientation;
	int bestMatchPoint;
	std::vector<double> orientation;
	std::vector<double> descriptors;
	double scale;

	Point() 
	{ 
		x = -1, y = -1, imageID = -1, bestMatchPoint = -1; 
		octavelevel = -1, interval = -1, maxOrientation = -1;
		scale = 0.0;
	}

};