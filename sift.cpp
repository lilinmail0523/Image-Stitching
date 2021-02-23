#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<vector>
#include <tuple>

#include "point.hpp"
#include "sift.hpp"

#define g_pi 3.1415927

#define g_peak_ratio_threshold 0.8

#define g_octave 4

#define g_intervals 2

#define g_contrast_threshold 0.02

#define g_curvature_threshold 10



// Generate the Gaussian pyramid/ Difference of Gaussian pyramid
void SIFT::generateDOG() {
	for (int oc = 0; oc < g_octave; oc++) {

		std::vector<double> siglist;
		double sigma = 1.6; // sqrt(3)
		double k = pow(2, (double)1 / g_intervals);
		// sigma list to build gaussians
		siglist.push_back(sigma);
		for (int i = 1; i < g_intervals + 3; i++) {
			double signext = sigma * k;
			siglist.push_back(sqrt(signext * signext - sigma * sigma));
			sigma = signext;
		}
		m_SigmaPyramids.push_back(siglist);



		// Gaussian pyramids
		std::vector<cv::Mat> GaussianImgs;
		GaussianImgs.push_back(m_ImgsPyramids[oc]);
		for (int i = 1; i < g_intervals + 3; i++) {
			cv::Mat blur;
			cv::GaussianBlur(m_ImgsPyramids[oc], blur, cv::Size(0, 0), siglist[i], siglist[i], cv::BORDER_REPLICATE);
			GaussianImgs.push_back(blur);
		}
		m_GaussianPyramids.push_back(GaussianImgs);



		//Differences of Gaussian pyramids
		std::vector<cv::Mat> DOG;
		for (int i = 1; i < g_intervals + 3; i++) {
			cv::Mat DiffGaussian;
			cv::subtract(GaussianImgs[i], GaussianImgs[i - 1], DiffGaussian, cv::noArray(), CV_64F);
			DOG.push_back(DiffGaussian);
			DiffGaussian.convertTo(DiffGaussian, CV_8U, 255.0);

		}
		m_DOGPyramids.push_back(DOG);

	 }


}


// find the local max/min in the 26 neighbors 
bool SIFT::findExtrema(int oct, int x, int y, int inls) {
	double localmin = m_DOGPyramids[oct][inls].at<double>(x, y);
	double localmax = m_DOGPyramids[oct][inls].at<double>(x, y);
	for (int inl = -1; inl <= 1; inl++) {
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				if (localmax < m_DOGPyramids[oct][inl + inls].at<double>(x + i, y + j)) {
					localmax = m_DOGPyramids[oct][inl + inls].at<double>(x + i, y + j);
				}
				if (localmin > m_DOGPyramids[oct][inl + inls].at<double>(x + i, y + j)) {
					localmin = m_DOGPyramids[oct][inl + inls].at<double>(x + i, y + j);
				}

			}
		}
	}



	if (localmin != m_DOGPyramids[oct][inls].at<double>(x, y) && localmax != m_DOGPyramids[oct][inls].at<double>(x, y)) {
		return false;
	}
	return true;

}

// first derivation to calculate the accurate location of feature points
cv::Mat SIFT::firstDerivation(int oct, double x, double y, int inl) {
	cv::Mat D = cv::Mat::zeros(3, 1, CV_64F);
	double dx = (m_DOGPyramids[oct][inl].at<double>(x + 1, y) - m_DOGPyramids[oct][inl].at<double>(x - 1, y)) / 2;
	double dy = (m_DOGPyramids[oct][inl].at<double>(x, y + 1) - m_DOGPyramids[oct][inl].at<double>(x, y - 1)) / 2;
	double ds = (m_DOGPyramids[oct][inl + 1].at<double>(x, y) - m_DOGPyramids[oct][inl - 1].at<double>(x, y)) / 2;

	D.at<double>(0, 0) = dx;
	D.at<double>(1, 0) = dy;
	D.at<double>(2, 0) = ds;

	return D;
}

// second derivation to calculate the accurate location of feature point and edge eliminating
cv::Mat SIFT::secondDerovation(int oct, double x, double y, int inl) {
	cv::Mat D = cv::Mat::zeros(3, 3, CV_64F);
	double dxx = m_DOGPyramids[oct][inl].at<double>(x + 1, y) + m_DOGPyramids[oct][inl].at<double>(x - 1, y) - 2 * m_DOGPyramids[oct][inl].at<double>(x, y);
	double dyy = m_DOGPyramids[oct][inl].at<double>(x, y + 1) + m_DOGPyramids[oct][inl].at<double>(x, y - 1) - 2 * m_DOGPyramids[oct][inl].at<double>(x, y);
	double dss = m_DOGPyramids[oct][inl + 1].at<double>(x, y) + m_DOGPyramids[oct][inl - 1].at<double>(x, y) - 2 * m_DOGPyramids[oct][inl].at<double>(x, y);
	double dxy = (m_DOGPyramids[oct][inl].at<double>(x + 1, y + 1) + m_DOGPyramids[oct][inl].at<double>(x - 1, y - 1) - m_DOGPyramids[oct][inl].at<double>(x + 1, y - 1) - m_DOGPyramids[oct][inl].at<double>(x - 1, y + 1)) / 4;
	double dxs = (m_DOGPyramids[oct][inl + 1].at<double>(x + 1, y) + m_DOGPyramids[oct][inl - 1].at<double>(x - 1, y) - m_DOGPyramids[oct][inl - 1].at<double>(x + 1, y) - m_DOGPyramids[oct][inl + 1].at<double>(x - 1, y)) / 4;
	double dys = (m_DOGPyramids[oct][inl + 1].at<double>(x, y + 1) + m_DOGPyramids[oct][inl - 1].at<double>(x, y - 1) - m_DOGPyramids[oct][inl + 1].at<double>(x, y - 1) - m_DOGPyramids[oct][inl - 1].at<double>(x, y + 1)) / 4;

	D.at<double>(0, 0) = dxx;
	D.at<double>(1, 0) = dxy;
	D.at<double>(2, 0) = dxs;
	D.at<double>(0, 2) = dxs;
	D.at<double>(1, 2) = dys;
	D.at<double>(2, 2) = dss;
	D.at<double>(0, 1) = dxy;
	D.at<double>(1, 1) = dyy;
	D.at<double>(2, 1) = dys;

	return D;
}

// find feature from DOGs
std::vector<std::vector<Point>> SIFT::findFeature() {
	std::vector<std::vector<Point>>features;


	for (int oc = 0; oc < g_octave; oc++) {
		std::vector<Point> ocfeatures;


		int border = m_DOGPyramids[oc].size() / 2 + 1;
		for (int img = 1; img < m_DOGPyramids[oc].size() - 1; img++) {
			for (int i = border; i < m_DOGPyramids[oc][img].rows - border; i++) {
				for (int j = border; j < m_DOGPyramids[oc][img].cols - border; j++) {

					// throw out low contrast 
					if (fabs(m_DOGPyramids[oc][img].at<double>(i, j)) <= g_contrast_threshold) {
						continue;
					}

					//throw out non-extremum
					if (!findExtrema(oc, i, j, img)) {
						continue;
					}

					// calculate offset
					cv::Mat fstDev = firstDerivation(oc, i, j, img);
					cv::Mat secDev = secondDerovation(oc, i, j, img);

					cv::Mat InvsecDev = secDev.inv();
					cv::Mat sft = -1 * InvsecDev * fstDev;


					//throw out low contrast
					if (fabs(0.5 * sft.dot(fstDev) + m_DOGPyramids[oc][img].at<double>(i, j) <= g_contrast_threshold)) {
						continue;
					}


					double Dxx = secDev.at<double>(0, 0), Dyy = secDev.at<double>(1, 1), Dxy = secDev.at<double>(1, 0);
					double Tr_H = Dxx + Dyy;
					double Det_H = Dxx * Dyy - Dxy * Dxy;
					double curvature_ratio = (Tr_H * Tr_H) / Det_H;


					double Threshold = ((g_curvature_threshold + 1) * (g_curvature_threshold + 1)) / g_curvature_threshold;

					//edge eliminating
					if (Det_H > 0 && curvature_ratio < Threshold) {
						Point pt;
						pt.x = i + static_cast<int>(sft.at<double>(0, 0));
						pt.y = j + static_cast<int>(sft.at<double>(1, 0));
						pt.interval = img + static_cast<int>(sft.at<double>(2, 0));
						pt.octavelevel = oc;
						pt.scale = pow(2, -oc + 1);

						ocfeatures.push_back(pt);
						//std::cout << pt.x << " " << pt.y << std::endl;
					}




				}
			}
		}

		features.push_back(ocfeatures);
	}
	return features;
}


std::vector<Point> SIFT::findOrientation(std::vector<std::vector<Point>> featurelist) {
	std::vector<Point> features;
	const int RADIUS = 8;
	for (int oc = 0; oc < g_octave; oc++) {
		for (auto pt : featurelist[oc]) {
			if (pt.interval < 0 || pt.interval > m_GaussianPyramids[oc].size() - 1) {
				continue;
			}

			cv::Mat GaussanImg = m_GaussianPyramids[oc][pt.interval];

			if (pt.x < (RADIUS + 1) || pt.y < (RADIUS + 1) || pt.x >= GaussanImg.rows - (RADIUS + 1) || pt.y >= GaussanImg.cols - (RADIUS + 1)) {
				continue;
			}

			double hist_orient[36] = { 0.0 };

			// calculate orientation and magnitude
			cv::Mat magnitudes = cv::Mat::zeros(RADIUS * 2, RADIUS * 2, CV_64F);
			cv::Mat orientations = cv::Mat::zeros(RADIUS * 2, RADIUS * 2, CV_64F);
			for (int i = pt.x - RADIUS; i < pt.x + RADIUS; i++) {
				for (int j = pt.y - RADIUS; j < pt.y + RADIUS; j++) {
					double dx = GaussanImg.at<double>(i + 1, j) - GaussanImg.at<double>(i - 1, j);
					double dy = GaussanImg.at<double>(i, j + 1) - GaussanImg.at<double>(i, j - 1);
					magnitudes.at<double>(i - (pt.x - RADIUS), j - (pt.y - RADIUS)) = sqrt(dx * dx + dy * dy);
					double ori = atan2(dx, dy) * 180 / g_pi;
					if (ori < 0) ori += 360;
					orientations.at<double>(i - (pt.x - RADIUS), j - (pt.y - RADIUS)) = ori;
				}
			}

			double sigma = m_SigmaPyramids[oc][pt.interval];


			cv::GaussianBlur(magnitudes, magnitudes, cv::Size(), sigma * pt.scale * 1.5, sigma * pt.scale * 1.5);
			for (int i = 0; i < RADIUS * 2; i++) {
				for (int j = 0; j < RADIUS * 2; j++) {
					int ori = static_cast<int>(orientations.at<double>(i, j) / 10);
					ori = ori % 36;
					hist_orient[ori] += magnitudes.at<double>(i, j);
				}

			}
			Point f;


			f.maxOrientation = 0;
			for (int i = 0; i < 36; i++) {
				if (hist_orient[f.maxOrientation] < hist_orient[i]) f.maxOrientation = hist_orient[i];
			}

			for (int i = 0; i < 36; i++) {
				if (hist_orient[i] > g_peak_ratio_threshold * f.maxOrientation) {
					f.orientation.push_back(i * 10 + 5);
				}
			}

			f.maxOrientation *= 10 + 5;
			f.x = pt.x;
			f.y = pt.y;
			f.imageID = m_SIFT_id;
			f.imageName = m_SIFT_filename;
			f.scale = pt.scale;
			f.interval = pt.interval;
			f.octavelevel = oc;
			features.push_back(f);

		}

	}
	return features;
}

std::vector<Point> SIFT::createDescriptor(std::vector<Point>& features) {
	const int RADIUS = 8;
	
	std::vector<Point> pts;

	for (Point pt : features) {

		if (pt.interval < 0 || pt.interval > m_GaussianPyramids[pt.octavelevel].size() - 1) {
			continue;
		}


		cv::Mat GaussanImg = m_GaussianPyramids[pt.octavelevel][pt.interval];

		if (pt.x < (RADIUS + 1) || pt.y < (RADIUS + 1) || pt.x >= GaussanImg.rows - (RADIUS + 1) || pt.y >= GaussanImg.cols - (RADIUS + 1)) {
			continue;
		}


		cv::Mat magnitudes = cv::Mat::zeros(RADIUS * 2, RADIUS * 2, CV_64F);
		cv::Mat orientations = cv::Mat::zeros(RADIUS * 2, RADIUS * 2, CV_64F);
		for (int i = pt.x - RADIUS; i < pt.x + RADIUS; i++) {
			for (int j = pt.y - RADIUS; j < pt.y + RADIUS; j++) {
				double dx = GaussanImg.at<double>(i + 1, j) - GaussanImg.at<double>(i - 1, j);
				double dy = GaussanImg.at<double>(i, j + 1) - GaussanImg.at<double>(i, j - 1);
				magnitudes.at<double>(i - (pt.x - RADIUS), j - (pt.y - RADIUS)) = sqrt(dx * dx + dy * dy);
				double ori = atan2(dx, dy) * 180 / g_pi;
				if (ori < 0) ori += 360;
				orientations.at<double>(i - (pt.x - RADIUS), j - (pt.y - RADIUS)) = ori;
			}
		}
		for (int i = 0; i < RADIUS * 2; i++) {
			for (int j = 0; j < RADIUS * 2; j++) {
				orientations.at<double>(i, j) += pt.maxOrientation;
			}
		}

		cv::GaussianBlur(magnitudes, magnitudes, cv::Size(), RADIUS, RADIUS);

		for (int i = 0; i < RADIUS * 2; i += 4) {
			for (int j = 0; j < RADIUS * 2; j += 4) {

				std::vector<double> hist_orient(8, 0);
				for (int x = 0; x < 4; x++) {
					for (int y = 0; y < 4; y++) {
						int ori = static_cast<int>((orientations.at<double>(i + x, j + y)) / 45);
						ori = ori % 8;
						hist_orient[ori] += magnitudes.at<double>(i + x, j + y);

					}
				}
				pt.descriptors.insert(pt.descriptors.end(), hist_orient.begin(), hist_orient.end());
			}
		}

		//std::cout << pt.descriptors.size() << std::endl;

		double sum = 0;
		for (double i : pt.descriptors) {
			sum += i * i;
		}
		double norm = sqrt(sum);

		for (int i = 0; i < pt.descriptors.size(); i++) {
			pt.descriptors[i] /= norm;
			if (pt.descriptors[i] < 0.2) pt.descriptors[i] = 0.2;
		}
		sum = 0;

		for (double i : pt.descriptors) {
			sum += i * i;
		}
		norm = sqrt(sum);
		for (int i = 0; i < pt.descriptors.size(); i++) {
			pt.descriptors[i] /= norm;
		}


		pt.x = (double)pt.x / pt.scale;
		pt.y = (double)pt.y / pt.scale;

		pts.push_back(pt);




	}

	return pts;


}


std::vector<Point> SIFT::SIFTProcessing(cv::Mat Img) {



	//convert into grayscale and normalize to [0,1]
	cv::Mat gray, normalize;
	cv::cvtColor(Img, gray, cv::COLOR_BGR2GRAY);
	gray.convertTo(normalize, CV_64F, 1.f / 255);




	//Size pyramid(Octave pyramid) build
	cv::Mat imgdoublesize;
	cv::resize(normalize, imgdoublesize, cv::Size(), 2, 2, cv::INTER_LINEAR);
	m_ImgsPyramids.push_back(imgdoublesize);
	for (int i = 1; i < g_octave; i++) {
		cv::Mat buf;
		cv::resize(m_ImgsPyramids[i - 1], buf, cv::Size(), 0.5, 0.5);
		m_ImgsPyramids.push_back(buf);
	}


	// Gaussian blur and Difference of Gaussian pyramid build
	generateDOG();


	// find feature point by DOG pyramid
	std::vector<std::vector<Point>> pts = findFeature();



	// find the orientation of features
	std::vector<Point> FeaturewithOrientation = findOrientation(pts);

	// calculate the descriptors of feature points 
	std::vector<Point> res = createDescriptor(FeaturewithOrientation);

	std::cout << m_SIFT_filename << " Total Point#: " << res.size() << std::endl;

	//plotPoint(Img, res);

	return res;
}

void SIFT::plotPoint(cv::Mat img, std::vector<Point>& Pts) {
	cv::Mat show = img.clone();
	int cnt = 0;
	for (Point pt : Pts) {

		cv::circle(show, cv::Point(pt.y, pt.x), 2, CV_RGB(255, 0, 0));
		cnt++;
	}


	std::string name = std::string(m_SIFT_filename.begin(), m_SIFT_filename.end() - 4) + "-pts" + std::to_string(cnt);
	cv::imwrite(name + ".png", show);
}




void test(std::vector<std::string> imglist) {

	std::cout << "--Starting SIFT--" << std::endl;
	std::vector<cv::Mat> Imgs;
	//grayscale image and normalize to 0-1
	for (std::string img : imglist) {
		Imgs.push_back(cv::imread(img, cv::IMREAD_UNCHANGED));
	}

	for (int i = 0; i < Imgs.size();i++) {
		std::cout << "Processing " << imglist[i] << std::endl;
		SIFT s{ imglist[i] , i };
		s.SIFTProcessing(Imgs[i]);
	}


}
