#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 1. Load two images (change paths to your KITTI images)
    Mat img_1 = imread("/media/sid/PortableSSD/data_odometry_gray/dataset/sequences/00/image_0/000000.png", IMREAD_GRAYSCALE);
    Mat img_2 = imread("/media/sid/PortableSSD/data_odometry_gray/dataset/sequences/00/image_0/000001.png", IMREAD_GRAYSCALE);

    if (img_1.empty() || img_2.empty()) {
        cout << "Error loading images" << endl;
        return -1;
    }

    // 2. Define keypoints and descriptors
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

    // 3. Create the ORB Detector
    Ptr<FeatureDetector> detector = ORB::create();
    
    // --- YOUR TASK STARTS HERE ---
    
    // Step A: Detect AND Compute
    // Instead of just detector->detect(), use detector->detectAndCompute()
    // It takes the image, a mask (noArray()), and outputs keypoints AND descriptors.
    detector->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
    detector->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);

    // Step B: Match the descriptors
    // Since ORB uses binary descriptors, we use Hamming distance, not Euclidean.
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);


// --- FILTERING STEP ---

    // 1. Find the minimum distance (best match)
    double min_dist = 10000;
    double max_dist = 0;

    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // 2. Keep only "good" matches (Low distance)
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        // The rule: Distance must be less than 2 * min_dist (or 30.0)
        if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }


    // Update the draw function to use 'good_matches' instead of 'matches'
    Mat img_good_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_good_matches);
    imshow("Good Matches", img_good_matches);
    // Step C: Draw the matches
    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);

    // --- YOUR TASK ENDS HERE ---

    imshow("ORB Matches", img_matches);

    // --- POSE ESTIMATION STEP ---

    // 1. Convert DMatch to Point2f
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < good_matches.size(); i++) {
        // queryIdx is the index in keypoints_1
        points1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
        // trainIdx is the index in keypoints_2
        points2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    // 2. Calculate Essential Matrix (This runs RANSAC internally)
    // Focal length (f) and Principal Point (pp) are from KITTI calibration
    double focal_length = 718.8560; 
    Point2d pp(607.1928, 185.2157);

    Mat E, R, t, mask;
    // The 'mask' will tell us which points were valid (Inliers)
    E = findEssentialMat(points1, points2, focal_length, pp, RANSAC, 0.999, 1.0, mask);

    // 3. Recover Pose (Rotation and Translation)
    recoverPose(E, points1, points2, R, t, focal_length, pp, mask);

    // Output the result
    cout << "Rotation Matrix (R):\n" << R << endl;
    cout << "Translation Vector (t):\n" << t << endl;
    
    waitKey(0);

    return 0;
}