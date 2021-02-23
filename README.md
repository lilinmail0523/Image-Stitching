# Image-Stitching
 
This is a project from [VFX course](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/18spring/assignments/proj2/), The image stitching is a process of combining serials of images with overlapping fields into a blending result like panoramas. The implementation included feature points detection by SIFT descriptors, image matching by KNN (from [ANN library](http://www.cs.umd.edu/~mount/ANN/)), and finally blending by alpha, and multiband blending. The OPENCV, ANN libraries were used in this project. 

## Input/Output:
Input: Image_list.txt with sequence images filename and corresponding focus length which could be acquired from autostitch

    Example: image_list.txt
    denny01.jpg 656.801
    denny02.jpg 660.261
    denny03.jpg 664.862
    denny04.jpg 669.626
    denny05.jpg 668.762
    denny06.jpg 646.531

Output: A panorama

## Feature detection
Feature points contain local information that can help us recognize their corresponding in multiples images. To match the corresponding, the robust matching needs high accuracy of the feature and invariance to affine changes. SIFT (scale-invariant feature transform) published by David lowe is a feature detection algorithm that takes advantage of difference-of-Gaussian pyramid to acquire more stable features that it is invariant to uniform scaling. The details of SIFT implementation were referenced from this [matlab code](http://ftp.cs.toronto.edu/pub/jepson/teaching/vision/2503/SIFTtutorial.zip).

<p align="center">
<img src="https://github.com/lilinmail0523/Image-Stitching/blob/main/results/denny.png" width="40%" height="40%" />
</p>
<p align="center">
Feature points in different scale of the image.
</p>

## Feature Matching

The aim of feature matching is to find closet features of two images for further stitching process. Kd-tree from ANN libraries was used to speed up the matching process by calculating the minimum Euclidean distance of two descriptors. For a key point, if the distance of closest feature < 0.8 * distance of second closest feature, it was considered as a match. After feature matching, the images and feature points were turned into cylindrical coordinates. There were some mismatches after feature matching because of similar local information in different regions. To deal with those outliers, the RANSAC was applied to calculate the pairwise alignment. The error for computing alignments was ![equation](https://github.com/lilinmail0523/Image-Stitching/blob/main/results/equation.jpg) 
, and m was the alignment of the coordinates. The RANSAC was run 50 iterations, and in every turn two of random points were selected to build the fitting model. Then the inlier group was set up by testing other data with a threshold,  and alignments were chosen by the minimum error of inlier groups.

## Blending

After matching and pairwise alignment, finally images were combined into the panorama. To cope with the overlapping region, the following methods were used:

<p align="center">
<img src="https://github.com/lilinmail0523/Image-Stitching/blob/main/results/Overlapping.png" width="40%" height="40%" />
</p>
<p align="center">
The overlapping region of two images.
</p>

1.	Direct connection: The overlapping region was divided vertically in half, and pasted by the overlapping region of two images which owned the half that was closer to it. The results showed that there were some visible seams.

<p align="center">
<img src="https://github.com/lilinmail0523/Image-Stitching/blob/main/results/dennyDirectBlending.png" width="50%" height="50%" />
</p>
<p align="center">
<img src="https://github.com/lilinmail0523/Image-Stitching/blob/main/results/prtnDirectblending.png" width="50%" height="50%" />
</p>
<p align="center">
The results of direct connection.
</p>

2.	Alpha blending: The overlapping region was set as the windows of blending. The weight was defined as the ratio of distance to the images, and the one that was closer to the images had the larger weight. The results showed that the visible seams were less than direct blending, but the results accompanied the blurred called ghost effects in the overlapping region.

<p align="center">
<img src="https://github.com/lilinmail0523/Image-Stitching/blob/main/results/dennyAlphablending.png" width="50%" height="50%" />
</p>
<p align="center">
<img src="https://github.com/lilinmail0523/Image-Stitching/blob/main/results/prtnAlphablending.png" width="50%" height="50%" />
</p>
<p align="center">
The results of alpha blending.
</p>

3.	Multiband blending: First, the Laplacian pyramid of two images was built, and then the Laplacian images were blended by using a weighted mask blurred from a Gaussian filter. Compared to alpha blending, there was no ghost effect but it remained some seams in the overlapping region. 

<p align="center">
<img src="https://github.com/lilinmail0523/Image-Stitching/blob/main/results/dennyMultibandblending.png" width="50%" height="50%" />
</p>
<p align="center">
<img src="https://github.com/lilinmail0523/Image-Stitching/blob/main/results/prtnMultibandblending.png" width="50%" height="50%" />
</p>
<p align="center">
The results of multiband blending.
</p>

## Reference
- [Distinctive Image Features from Scale-Invariant Keypoints, David G. Lowe, 2004](https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)
- [Recognising Panoramas, by M.Brown and D. G. Lowe, 2003.](http://matthewalunbrown.com/papers/iccv2003.pdf)
