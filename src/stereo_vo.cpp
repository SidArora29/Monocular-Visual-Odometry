#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

// KITTI Calibration Data (Sequence 00-02)
const double focal = 718.8560;
const cv::Point2d pp(607.1928, 185.2157);
const double baseline = 0.537; // Baseline in meters

int main() {
    // ---------------- INITIALIZATION ----------------
    Mat R_f = Mat::eye(3, 3, CV_64F);
    Mat t_f = Mat::zeros(3, 1, CV_64F);
    
    // Visualization window
    Mat traj = Mat::zeros(600, 600, CV_8UC3);

    // Load the very first frame (Left and Right)
    // UPDATE THESE PATHS TO YOUR SYSTEM
    string path_left = "/media/sid/PortableSSD/data_odometry_gray/dataset/sequences/00/image_0/";
    string path_right = "/media/sid/PortableSSD/data_odometry_gray/dataset/sequences/00/image_1/";

    char filename1[200], filename2[200];
    sprintf(filename1, "%s%06d.png", path_left.c_str(), 0);
    sprintf(filename2, "%s%06d.png", path_right.c_str(), 0);

    Mat img_left_prev = imread(filename1, IMREAD_GRAYSCALE);
    Mat img_right_prev = imread(filename2, IMREAD_GRAYSCALE);

    if (img_left_prev.empty() || img_right_prev.empty()) {
        cout << "Error: Could not load initial images!" << endl;
        return -1;
    }

    // Detect initial features in the first Left image
    vector<Point2f> points_left_prev;
    goodFeaturesToTrack(img_left_prev, points_left_prev, 2000, 0.01, 10);

    // ---------------- MAIN LOOP ----------------
    for (int numFrame = 1; numFrame < 1000; numFrame++) {
        
        // Load Current Frame (Left Only)
        sprintf(filename1, "%s%06d.png", path_left.c_str(), numFrame);
        Mat img_left_curr = imread(filename1, IMREAD_GRAYSCALE);

        if (img_left_curr.empty()) break;

        // --- STEP 1: CALCULATE DISPARITY (DEPTH) FOR PREVIOUS FRAME ---
        // We match points from Left_Prev -> Right_Prev to get Z
        
        vector<Point2f> points_right_prev;
        vector<uchar> status_stereo;
        vector<float> err_stereo;

        // Use Optical Flow to find the match in the Right image (faster than block matching)
        calcOpticalFlowPyrLK(img_left_prev, img_right_prev, points_left_prev, points_right_prev, status_stereo, err_stereo, Size(21, 21), 3);

        // --- STEP 2: TRACK POINTS TO CURRENT FRAME ---
        // We match points from Left_Prev -> Left_Curr to get Motion
        vector<Point2f> points_left_curr;
        vector<uchar> status_track;
        vector<float> err_track;

        calcOpticalFlowPyrLK(img_left_prev, img_left_curr, points_left_prev, points_left_curr, status_track, err_track, Size(21, 21), 3);

        // --- STEP 3: FILTER POINTS & PREPARE FOR PnP ---
        vector<Point3f> objectPoints; // 3D World Points (Relative to Prev Camera)
        vector<Point2f> imagePoints;  // 2D Image Points (in Current Camera)
        vector<Point2f> valid_points_left_curr; // Points to keep for next iteration

        for (size_t i = 0; i < points_left_prev.size(); i++) {
            // Check if point was found in BOTH Stereo match and Time track
            if (status_stereo[i] && status_track[i]) {
                
                // Calculate Disparity: (Left X - Right X)
                float d = points_left_prev[i].x - points_right_prev[i].x;

                // Filter invalid disparities
                if (d < 0.1) continue; // Close to infinity or negative

                // Calculate Depth (Z) = (f * B) / d
                float z = (focal * baseline) / d;

                // Filter outliers (Too close or too far)
                if (z < 0.5 || z > 50.0) continue;

                // Reconstruct 3D Point (X, Y, Z) relative to Previous Camera
                float x = (points_left_prev[i].x - pp.x) * z / focal;
                float y = (points_left_prev[i].y - pp.y) * z / focal;

                objectPoints.push_back(Point3f(x, y, z));
                imagePoints.push_back(points_left_curr[i]);
                valid_points_left_curr.push_back(points_left_curr[i]);
            }
        }

        // --- STEP 4: ESTIMATE MOTION (PnP) ---
        if (objectPoints.size() < 10) {
            cout << "Lost tracking, re-initializing..." << endl;
             goodFeaturesToTrack(img_left_prev, points_left_prev, 2000, 0.01, 10);
             continue;
        }

        Mat rvec, tvec, R;
        // Solve PnP: Finds rotation/translation that minimizes reprojection error

        Mat K = (Mat_<double>(3,3) << focal, 0, pp.x, 
                                      0, focal, pp.y, 
                                      0, 0, 1);

        solvePnPRansac(objectPoints, imagePoints, K, noArray(), rvec, tvec);
        Rodrigues(rvec, R); // Convert rotation vector to matrix

        // --- STEP 5: UPDATE TRAJECTORY ---
        // PnP gives T_curr_from_prev (World points -> Current Camera)
        // We need the Camera Position in World Frame.
        
        // PnP equation: P_curr = R * P_prev + t
        // We want to accumulate the global pose.
        // t_local_movement = -R_transposed * t
        Mat t_local = -R.t() * tvec;

        // Global Update:
        // t_final = t_final + (R_final * t_local)
        // R_final = R_final * R_transposed
        
        t_f = t_f + (R_f * t_local);
        R_f = R_f * R.t();

        // --- DRAWING ---
        // Scale and shift for visualization (No "scale cheat" needed anymore!)
        // Initial x,z are at 0,0. Let's center on image.
        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) ; // Using Z as Y for 2D plotting

        circle(traj, Point(x, y), 1, Scalar(0, 255, 0), 2);
        
        imshow("Stereo VO Trajectory", traj);
        imshow("Camera View", img_left_curr);
        
        char key = waitKey(1);
        if (key == 27) break; // Esc to quit

        // --- PREPARE FOR NEXT ITERATION ---
        // 1. Current Left Image becomes Previous Left Image
        img_left_prev = img_left_curr.clone();
        
        // 2. Load the Right Image for the next iteration (Current becomes Prev)
        sprintf(filename2, "%s%06d.png", path_right.c_str(), numFrame);
        img_right_prev = imread(filename2, IMREAD_GRAYSCALE);

        // 3. Keep tracking the good points
        // If we drop below a threshold, find new features
        if (valid_points_left_curr.size() < 1000) {
            goodFeaturesToTrack(img_left_prev, points_left_prev, 2000, 0.01, 10);
        } else {
            points_left_prev = valid_points_left_curr;
        }

        // --- SAVING THE MAP (OPTIONAL) ---
        // Save every 5th frame to avoid making a 1GB file
        if (numFrame % 5 == 0) {
            ofstream f;
            f.open("stereo_map23.xyz", ios::app);
            for (Point3f p : objectPoints) {
                // Important: These points are relative to the PREVIOUS camera.
                // We need to rotate them to the GLOBAL frame.
                // X_global = R_final * X_local + t_final
                
                Mat p_mat = (Mat_<double>(3,1) << p.x, p.y, p.z);
                Mat p_global = R_f * p_mat + t_f;
                
                // Save X, Y, Z
                f << p_global.at<double>(0) << " " << p_global.at<double>(1) << " " << p_global.at<double>(2) << endl;
            }
            f.close();
        }
    }

    return 0;
}