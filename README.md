# Image-Stitching
 
This is a project from [VFX course](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/18spring/assignments/proj2/), The image stitching is a process of combining serials of images with overlapping fields into a panorama. The implementation included following steps:
1. Feature-based registration by Scale-invariant feature transform (SIFT)
2. Feature matching by finding nearest neighbors (done by *k*-d trees from [ANN library](http://www.cs.umd.edu/~mount/ANN/))
3. Alignment calculation by least squares and RANSAC
4. Blending by direction connection, alpha blending, and Laplacian pyramid blending. 
 
The details of the implementation and the comparison of blending methods were in the [report.pdf](https://github.com/lilinmail0523/Image-Stitching/blob/main/report.pdf). The OPENCV, ANN libraries were used in this project. 

## Input/Output:
Input: Image_list.txt with sequence images and corresponding focus length which could be acquired from autostitch

    Example: image_list.txt
    denny01.jpg 656.801
    denny02.jpg 660.261
    denny03.jpg 664.862
    denny04.jpg 669.626
    denny05.jpg 668.762
    denny06.jpg 646.531

Output: A panorama


## Results (Laplacian pyramid blending)

<p align="center">
<img src="https://github.com/lilinmail0523/Image-Stitching/blob/main/results/dennyMultibandblending.png">
</p>
<p align="center">
<img src="https://github.com/lilinmail0523/Image-Stitching/blob/main/results/prtnMultibandblending.png">
</p>

## Reference
- [Distinctive Image Features from Scale-Invariant Keypoints, David G. Lowe, 2004](https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)
- [Recognising Panoramas, by M.Brown and D. G. Lowe, 2003.](http://matthewalunbrown.com/papers/iccv2003.pdf)
- [Image Alignment and Stitching: A Tutorial, Richard Szeliski, 2006](https://www.microsoft.com/en-us/research/wp-content/uploads/2004/10/tr-2004-92.pdf)
