#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<vector>
#include <fstream>
#include <tuple>

#include<time.h>
#include<ANN.h>


#include "point.hpp"
#include "sift.hpp"
#include "featurematching.hpp"
#include "blending.hpp"






int main() {
	std::string Imgfile;
	std::ifstream  Imagelist("image_list.txt");
	std::vector<std::string> imglist;
	std::vector<double> focallist;

	if (!Imagelist) {
		std::cout << " Image List isn't loaded!" << std::endl;
		return 0;
	}
	else {
		while (std::getline(Imagelist, Imgfile)) {


			std::istringstream iss(Imgfile);
			std::string s;
			//img file name
			iss >> s;
			imglist.push_back(s);

			// img focal length
			iss >> s;
			focallist.push_back(stod(s));
		}
	}
	Imagelist.close();


	std::vector<std::vector<Point>> Features;
	/*Building SIFT points*/
	for (int i = 0; i < imglist.size(); i++) {
		SIFT s{ imglist[i] , i };
		Features.push_back(s.SIFTProcessing(cv::imread(imglist[i], cv::IMREAD_UNCHANGED)));
	}



	/*Cylindrical projection of images and sift feature points*/
	std::vector<cv::Mat> CylindricalImgs;
	for (int i = 0; i < imglist.size(); i++) {
		CylindricalImgs.push_back(cylindricalProjection(cv::imread(imglist[i], -1), focallist[i], Features[i]));
	}

	// check the blending sequence 
	bool backward = false;

	std::vector<std::pair<int, int>> moves;
	// Compute the pairwise alignments of pairs of images
	for (int i = 1; i < CylindricalImgs.size(); i++) {
		std::vector<Point> f11, f22;
		std::tie(f11, f22) = KNNMatching(Features[i-1], Features[i]);
		//plotMatching(f11, f22, CylindricalImgs);
		std::pair<int, int> move = calculateShift(f11, f22, CylindricalImgs[i-1].cols);
		moves.push_back(move);
		if (move.second > CylindricalImgs[i].cols) backward = true;
	}


	// check the blending sequence 
	//backward = true ==> from left to right
	if (backward) {
		std::reverse(CylindricalImgs.begin(), CylindricalImgs.end());
		std::reverse(moves.begin(), moves.end());
		for (int i = 0; i < moves.size(); i++) {
			moves[i].first = -moves[i].first;
			moves[i].second = CylindricalImgs[i].cols + CylindricalImgs[i + 1].cols - moves[i].second;
		}
	}


	cv::Mat res = CylindricalImgs[0].clone();



	for (int i = 0; i < moves.size(); i++) {
		
		//expand border to blend
		//rows expension depend on the shift orientation
		if (moves[i].first > 0) {
			cv::copyMakeBorder(res, res, moves[i].first, 0, 0, CylindricalImgs[i + 1].cols - moves[i].second, cv::BORDER_CONSTANT, 0);
			//row shift
			cv::copyMakeBorder(CylindricalImgs[i + 1], CylindricalImgs[i + 1], 0, moves[i].first, 0, 0, cv::BORDER_CONSTANT, 0);

		}
		else {
			cv::copyMakeBorder(res, res, 0, abs(moves[i].first), 0, CylindricalImgs[i + 1].cols - moves[i].second, cv::BORDER_CONSTANT, 0);
			//row shift
			cv::copyMakeBorder(CylindricalImgs[i + 1], CylindricalImgs[i + 1], abs(moves[i].first), 0, 0, 0, cv::BORDER_CONSTANT, 0);

		}

		multibandBlending(res, CylindricalImgs[i+1], moves[i].first, moves[i].second);



	}

	// > 2 images, crop image with average height, same height of original images, so average height = calculate the average of shifts
	if (moves.size() > 1) {
		int MoveAverage = 0, top = 0, bottom = 0;
		for (auto move : moves) {
			MoveAverage += move.first;
			if (move.first >= 0) top += move.first;
		}

		cv::Rect Cropwindow(0, top - MoveAverage / (moves.size() - 1), res.cols, top + CylindricalImgs[0].cols - MoveAverage / (moves.size() - 1));
		cv::imwrite("result.png", res(Cropwindow));
	}

	// only two image, label the overlapping region for presentation
	else {
		cv::Rect overlappingregion(res.cols - CylindricalImgs[1].cols, 0, moves[0].second, CylindricalImgs[1].rows);
		cv::rectangle(res, overlappingregion, CV_RGB(0, 255, 0), 5);
		cv::imwrite("result.png", res);

	}




	return 0;
}