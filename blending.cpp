#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<time.h>



#include "point.hpp"
#include "blending.hpp"

cv::Mat cylindricalProjection(cv::Mat img, double f, std::vector<Point>& pts) {
	cv::Mat warped = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int x = round(f * (i - img.rows / 2) / sqrt((j - img.cols / 2) * (j - img.cols / 2) + f * f)) + img.rows / 2;
			int y = round(f * atan((j - img.cols / 2) / f) + img.cols / 2);


			warped.at<cv::Vec3b>(x, y)[0] = img.at<cv::Vec3b>(i, j)[0];
			warped.at<cv::Vec3b>(x, y)[1] = img.at<cv::Vec3b>(i, j)[1];
			warped.at<cv::Vec3b>(x, y)[2] = img.at<cv::Vec3b>(i, j)[2];

		}
	}





	int y0 = round(f * atan((0 - img.cols / 2) / f) + img.cols / 2);
	int ymax = round(f * atan((img.cols - img.cols / 2) / f) + img.cols / 2);

	cv::Rect rectCrop(cv::Point(y0, 0), cv::Point(ymax, img.rows));
	cv::Mat res = warped(rectCrop);



	for (Point& pt : pts) {
		int x = round(f * (pt.x - img.rows / 2) / sqrt((pt.y - img.cols / 2) * (pt.y - img.cols / 2) + f * f)) + img.rows / 2;
		int y = round(f * atan((pt.y - img.cols / 2) / f) + img.cols / 2);
		pt.x = x;
		pt.y = y - y0;


	}



	cv::waitKey();
	return res;
}

std::pair<int, int> calculateShift(std::vector<Point>& ptsLeft, std::vector<Point>& ptsRight, int prev_width, double threshold, int groupsize) {
	srand(time(NULL));
	int minerror = INT_MAX, min_x = 0, min_y = 0, matchsize = 0;
	std::cout << "Pairwise Alignment of Image " << ptsLeft[0].imageName << " and " << ptsRight[0].imageName << std::endl;

	int iter = 0;
	while (iter < 50 || matchsize == 0) {

		int randompoint = rand() % ptsLeft.size();
		int mx = (ptsRight[randompoint].x) - ptsLeft[randompoint].x;
		int my = (ptsRight[randompoint].y + prev_width) - ptsLeft[randompoint].y;

		std::vector<int> alsoInliers;

		int err = 0;
		for (int i = 0; i < ptsRight.size(); i++) {

			int dx = (mx + ptsLeft[i].x) - ptsRight[i].x;
			int dy = (my + ptsLeft[i].y) - (ptsRight[i].y + prev_width);
			err = sqrt(dx * dx + dy * dy);
			if (i != randompoint && err < threshold) alsoInliers.push_back(i);
		}





		//if the inlier group size is too small --> skip
		if (alsoInliers.size() >= groupsize) {
			err = 0;
			for (int j : alsoInliers) {
				int dx = (mx + ptsLeft[j].x) - ptsRight[j].x;
				int dy = (my + ptsLeft[j].y) - (ptsRight[j].y + prev_width);
				err += sqrt(dx * dx + dy * dy);

			}

			if (err < minerror) {
				minerror = err;
				min_x = mx;
				min_y = my;
				matchsize = alsoInliers.size();
			}
		}


		iter++;

		/*
			if the alsoInliers group is too small to build an inlier group in 50 iterations, 
			adjust the the threshold to fit the pairwise alignment
		*/
		if (iter == 50) {
			if (matchsize == 0) {
				iter = 0;
				threshold += 5;
				std::cout << "alsoInlier group is small threshlod++" << std::endl;
			}
		}

	}
	std::cout << "minx = " << min_x << std::endl;
	std::cout << "miny = " << min_y << std::endl;
	std::cout << "Inlier group size = " << matchsize << std::endl;

	return { min_x, min_y };

}

void alphaBlending(cv::Mat &Img, cv::Mat ImagetoBlend, int move_x, int move_y, bool DirectConnection) {

	for (int i = 0; i < ImagetoBlend.rows; i++) {
		for (int j = 0; j < ImagetoBlend.cols; j++) {
			int x = i;
			int y = j + (Img.cols - ImagetoBlend.cols);
			/*previous images region*/
			if (ImagetoBlend.at<cv::Vec3b>(i, j)[0] == 0 && ImagetoBlend.at<cv::Vec3b>(i, j)[1] == 0 && ImagetoBlend.at<cv::Vec3b>(i, j)[2] == 0) {
				continue;
			}

			/*the latter imgages region*/
			else if (Img.at<cv::Vec3b>(x, y)[0] == 0 && Img.at<cv::Vec3b>(x, y)[1] == 0 && Img.at<cv::Vec3b>(x, y)[2] == 0) {
				Img.at<cv::Vec3b>(x, y)[0] = ImagetoBlend.at<cv::Vec3b>(i, j)[0];
				Img.at<cv::Vec3b>(x, y)[1] = ImagetoBlend.at<cv::Vec3b>(i, j)[1];
				Img.at<cv::Vec3b>(x, y)[2] = ImagetoBlend.at<cv::Vec3b>(i, j)[2];
			}

			/*overlap region*/
			else {
				double alpha = (double)(j) / move_y;
				if (DirectConnection == true) {
					if (j <= move_y/2) alpha = 0;
					else alpha = 1;
				}
				Img.at<cv::Vec3b>(x, y)[0] = (int)Img.at<cv::Vec3b>(x, y)[0] * (1 - alpha) + ImagetoBlend.at<cv::Vec3b>(i, j)[0] * ( alpha);
				Img.at<cv::Vec3b>(x, y)[1] = (int)Img.at<cv::Vec3b>(x, y)[1] * (1 - alpha) + ImagetoBlend.at<cv::Vec3b>(i, j)[1] * ( alpha);
				Img.at<cv::Vec3b>(x, y)[2] = (int)Img.at<cv::Vec3b>(x, y)[2] * (1 - alpha) + ImagetoBlend.at<cv::Vec3b>(i, j)[2] * ( alpha);


			}
		}
	}
}



std::vector<cv::Mat> gaussianPyramid(cv::Mat img, int leveln) {
	std::vector<cv::Mat> GaussianImages;
	GaussianImages.push_back(img);

	for (int i = 0; i < leveln; i++) {
		cv::Mat temp;
		cv::pyrDown(GaussianImages[i], temp);
		GaussianImages.push_back(temp);
	}

	return GaussianImages;
}


std::vector<cv::Mat> LaplacianPyramid(std::vector<cv::Mat>& GaussianImages, int leveln) {
	std::vector<cv::Mat> LaplacianImages;

	LaplacianImages.push_back(GaussianImages.back());
	for (int i = leveln; i > 0; i--) {
		cv::Mat temp, L;


		cv::pyrUp(GaussianImages[i], temp, cv::Size(GaussianImages[i - 1].cols, GaussianImages[i - 1].rows));

		cv::subtract(GaussianImages[i - 1], temp, L);

		LaplacianImages.push_back(L);
	}

	return LaplacianImages;
}

void multibandBlending(cv::Mat &Img, cv::Mat ImagetoBlend, int move_x, int move_y, int leveln) {
	//crop the overlap region
	cv::Rect windowA(Img.cols - ImagetoBlend.cols, 0, move_y, ImagetoBlend.rows);
	cv::Rect windowB(0, 0, move_y, ImagetoBlend.rows);


	cv::Mat subA; Img(windowA).convertTo(subA, CV_32FC3, 1 / 255.0);
	cv::Mat subB; ImagetoBlend(windowB).convertTo(subB, CV_32FC3, 1 / 255.0);
	
	// build the weight mask
	cv::Mat mask = cv::Mat::zeros(Img.rows, move_y, CV_32F);

	for (int x = 0; x < mask.rows; x++) {
		for (int y = 0; y < mask.cols; y++) {
			if (y < (mask.cols / 2))
				mask.at < float >(x, y) = 1.0;
		}
	}

	std::vector <cv::Mat> GaussianSubA = gaussianPyramid(subA, leveln);
	std::vector <cv::Mat> GaussianSubB = gaussianPyramid(subB, leveln);
	std::vector <cv::Mat> GaussianMask = gaussianPyramid(mask, leveln);


	std::vector <cv::Mat> LaplacianSubA = LaplacianPyramid(GaussianSubA, leveln);
	std::vector <cv::Mat> LaplacianSubB = LaplacianPyramid(GaussianSubB, leveln);

	std::vector<cv::Mat> concatenatedImage;
	for (int i = 0; i <= leveln; i++) {
		cv::Mat combine = cv::Mat::zeros(LaplacianSubA[i].rows, LaplacianSubA[i].cols, CV_32FC3);
		for (int x = 0; x < LaplacianSubA[i].rows; x++) {
			for (int y = 0; y < LaplacianSubA[i].cols; y++) {
				float alpha = GaussianMask[leveln - i].at<float>(x, y);
				combine.at<cv::Vec3f>(x, y)[0] = (LaplacianSubA[i].at<cv::Vec3f>(x, y)[0] * alpha + LaplacianSubB[i].at<cv::Vec3f>(x, y)[0] * (1 - alpha));
				combine.at<cv::Vec3f>(x, y)[1] = (LaplacianSubA[i].at<cv::Vec3f>(x, y)[1] * alpha + LaplacianSubB[i].at<cv::Vec3f>(x, y)[1] * (1 - alpha));
				combine.at<cv::Vec3f>(x, y)[2] = (LaplacianSubA[i].at<cv::Vec3f>(x, y)[2] * alpha + LaplacianSubB[i].at<cv::Vec3f>(x, y)[2] * (1 - alpha));

			}
		}

		concatenatedImage.push_back(combine);
	}


	cv::Mat blend = concatenatedImage[0];
	for (int i = 1; i <= leveln; i++) {
		cv::Mat conc = concatenatedImage[i];
		cv::pyrUp(blend, blend, cv::Size(concatenatedImage[i].cols, concatenatedImage[i].rows));

		cv::add(blend, concatenatedImage[i], blend);

	}
	blend.convertTo(blend, CV_8UC3, 255.0);

	/*
	cv::imshow("subA", blend);
	cv::waitKey();
	*/

	for (int i = 0; i < ImagetoBlend.rows; i++) {
		for (int j = 0; j < ImagetoBlend.cols; j++) {
			int x = i;
			int y = j + (Img.cols - ImagetoBlend.cols);
			/*previous images region*/
			if (ImagetoBlend.at<cv::Vec3b>(i, j)[0] == 0 && ImagetoBlend.at<cv::Vec3b>(i, j)[1] == 0 && ImagetoBlend.at<cv::Vec3b>(i, j)[2] == 0) {
				continue;
			}

			/*the latter imgages region*/
			else if (Img.at<cv::Vec3b>(x, y)[0] == 0 && Img.at<cv::Vec3b>(x, y)[1] == 0 && Img.at<cv::Vec3b>(x, y)[2] == 0) {
				Img.at<cv::Vec3b>(x, y)[0] = ImagetoBlend.at<cv::Vec3b>(i, j)[0];
				Img.at<cv::Vec3b>(x, y)[1] = ImagetoBlend.at<cv::Vec3b>(i, j)[1];
				Img.at<cv::Vec3b>(x, y)[2] = ImagetoBlend.at<cv::Vec3b>(i, j)[2];
			}


			else {
				Img.at<cv::Vec3b>(x, y)[0] = blend.at<cv::Vec3b>(i, j)[0];
				Img.at<cv::Vec3b>(x, y)[1] = blend.at<cv::Vec3b>(i, j)[1];
				Img.at<cv::Vec3b>(x, y)[2] = blend.at<cv::Vec3b>(i, j)[2];
			}
		}

	}


}
