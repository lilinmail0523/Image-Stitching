#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<vector>
#include<tuple>

#include<ANN.h>
#include "point.hpp"
#include "featurematching.hpp"

void plotMatching(std::vector<Point> Pts1, std::vector<Point> Pts2, std::vector<cv::Mat> warpedimgs) {
	int imgLeftIndex = Pts1[0].imageID;
	int cols = warpedimgs[imgLeftIndex].cols;

	int imgRightIndex = Pts2[0].imageID;

	cv::Mat matchres;
	cv::hconcat(warpedimgs[imgLeftIndex], warpedimgs[imgRightIndex], matchres);
	int cnt = 0;
	for (int i = 0; i < Pts1.size(); i++) {
		Point pt = Pts1[i];
		if (pt.bestMatchPoint == -1) continue;

		int Matchindex = pt.bestMatchPoint;

		cnt++;
		cv::circle(matchres, cv::Point(pt.y , pt.x ), 2, CV_RGB(255, 0, 0));
		cv::circle(matchres, cv::Point(Pts2[i].y + cols, Pts2[i].x), 2, CV_RGB(255, 0, 0));
		cv::line(matchres, cv::Point(pt.y, pt.x ), cv::Point(Pts2[i].y + cols, Pts2[i].x), CV_RGB(255, 0, 0));
	}
	
	std::string leftName = std::string(Pts1[0].imageName.begin(), Pts1[0].imageName.end() - 4);
	std::string rightName = std::string(Pts2[0].imageName.begin(), Pts2[0].imageName.end() - 4);



	std::cout << cnt;
	cv::imwrite(leftName + rightName + ".png", matchres);
	cv::waitKey(0);
}


std::tuple <std::vector<Point>, std::vector<Point>> KNNMatching(std::vector<Point> &Pts1, std::vector<Point> &Pts2) {
	ANNpointArray pt1, pt2;
	ANNkd_tree* kdTree1 , * kdTree2 ;

	pt1 = annAllocPts(Pts1.size(), 128);
	pt2 = annAllocPts(Pts2.size(), 128);
	for (int i = 0; i < Pts1.size(); i++) {
		for (int feature = 0; feature < 128; feature++) {

			pt1[i][feature] = Pts1[i].descriptors[feature];
		}
	}
	kdTree1 = new ANNkd_tree(pt1, Pts1.size(),128);

	for (int i = 0; i < Pts2.size(); i++) {
		for (int feature = 0; feature < 128; feature++) {
			pt2[i][feature] = Pts2[i].descriptors[feature];
		}
	}
	kdTree2 = new ANNkd_tree(pt2, Pts2.size(), 128);

	ANNpoint queryPt = annAllocPt(128);
	ANNidxArray nnIdx = new ANNidx[2];
	ANNdistArray dists = new ANNdist[2];
	for (int i = 0; i < Pts1.size(); i++) {
		for (int feature = 0; feature < 128; feature++) {
			queryPt[feature] = Pts1[i].descriptors[feature];
		}

		kdTree2->annkSearch(queryPt, 2, nnIdx, dists);
		if (dists[0] < dists[1] * 0.8) {
			Pts1[i].bestMatchPoint = nnIdx[0];
		}
	}

	for (int i = 0; i < Pts2.size(); i++) {
		for (int feature = 0; feature < 128; feature++) {
			queryPt[feature] = Pts2[i].descriptors[feature];
		}

		kdTree1->annkSearch(queryPt, 2, nnIdx, dists);
		if (dists[0] < dists[1] * 0.8) {

			Pts2[i].bestMatchPoint = nnIdx[0];

		}
	}
	
	std::vector<Point> output1, output2;
	for (int i = 0; i < Pts1.size();i++) {
		/*remove the point that match nothing or wrong points*/
		int id = Pts1[i].bestMatchPoint;
		if (id == -1 || id >= Pts2.size() ) continue;
		if (Pts2[id].bestMatchPoint != i) {

			continue;
		}
		output1.push_back(Pts1[i]);
		output2.push_back(Pts2[id]);

	}
	
	delete[] dists;
	delete[] nnIdx;
	//delete kdTree1, 
	//delete kdTree2;
	annDeallocPt(queryPt);
	annClose();

	return { output1,output2 };

}

