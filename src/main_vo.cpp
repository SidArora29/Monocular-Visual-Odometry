#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

void savePoints(string filename, vector<Point3f> points) {
    ofstream f;
    f.open(filename, ios::app); // Open in append mode
    for (size_t i = 0; i < points.size(); i++) {
        // Save as: X Y Z (simple text format)
        f << points[i].x << " " << points[i].y << " " << points[i].z << endl;
    }
    f.close();
}

// Helper to convert OpenCV 4D matrix to 3D points
void mat2Points(Mat& points4D, vector<Point3f>& points3D) {
    for (int i = 0; i < points4D.cols; i++) {
        Mat col = points4D.col(i);
        // Divide by W to normalize (Homogeneous -> Cartesian)
        float w = col.at<float>(3); 
        float x = col.at<float>(0) / w;
        float y = col.at<float>(1) / w;
        float z = col.at<float>(2) / w;
        
        // Filter out points that are too far away or behind camera (junk)
        if (z > 0 && z < 50) { 
            points3D.push_back(Point3f(x, y, z));
        }
    }
}
// Helper to get absolute scale from ground truth (The "Cheat")
double getAbsoluteScale(int frame_id) {
    string line;
    int i = 0;
    ifstream myfile("/media/sid/PortableSSD/data_odometry_poses/dataset/poses/00.txt"); 
    double x = 0, y = 0, z = 0;
    double x_prev, y_prev, z_prev;
    
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            stringstream ss(line);
            ss >> x_prev >> x_prev >> x_prev >> x_prev >> x_prev >> x_prev >> x_prev >> x_prev >> x_prev >> x_prev >> x_prev; // Skip rotation
            ss >> x; // We only need the translation columns (3, 7, 11) but KITTI is 12 floats per row.
            // Actually simpler: standard KITTI pose format is r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
            // Let's just calculate distance between current row and prev row.
            // ... (For simplicity, let's just return 1.0 if you don't have the file yet, 
            // but the path will be scale-distorted)
            if (i == frame_id) return sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev));
            i++;
        }
        myfile.close();
    }
    return 1.0; // Default if file not found
}

// SIMPLIFIED VERSION: Just assumes constant speed if you don't have poses.txt handy
double getScale(int frame_id) {
    return 1.0; 
}

int main() {
    // Initialization
    Mat img_1, img_2;
    Mat R_f = Mat::eye(3, 3, CV_64F);
    Mat t_f = Mat::zeros(3, 1, CV_64F);
    
    // Create a blank map image to draw the path
    Mat traj = Mat::zeros(600, 600, CV_8UC3);

    // Camera Intrinsics
    double focal = 718.8560;
    Point2d pp(607.1928, 185.2157);

    // Loop through the first 1000 frames
    for (int numFrame = 0; numFrame < 1000; numFrame++) {
        // Load images (Update paths!)
        char filename1[200];
        char filename2[200];
        sprintf(filename1, "/media/sid/PortableSSD/data_odometry_gray/dataset/sequences/00/image_0/%06d.png", numFrame);
        sprintf(filename2, "/media/sid/PortableSSD/data_odometry_gray/dataset/sequences/00/image_0/%06d.png", numFrame + 1);

        img_1 = imread(filename1, IMREAD_GRAYSCALE);
        img_2 = imread(filename2, IMREAD_GRAYSCALE);

        if (img_1.empty() || img_2.empty()) {
            cout << "End of sequence!" << endl;
            break;
        }

        // --- YOUR CODE FROM BEFORE GOES HERE ---
        // 1. Detect (ORB)
        vector<KeyPoint> k1, k2;
        Mat d1, d2;
        Ptr<FeatureDetector> detector = ORB::create(3000);
        detector->detectAndCompute(img_1, noArray(), k1, d1);
        detector->detectAndCompute(img_2, noArray(), k2, d2);
        
        // 2. Match (BFMatcher + Filter)
        BFMatcher matcher(NORM_HAMMING);
        vector<DMatch> matches;
        matcher.match(d1, d2, matches);
        
        vector<DMatch> good_matches;
        double min_dist = 10000;
        for( int i = 0; i < d1.rows; i++ ) { if( matches[i].distance < min_dist ) min_dist = matches[i].distance; }
        for( int i = 0; i < d1.rows; i++ ) { if( matches[i].distance <= max(2*min_dist, 30.0) ) good_matches.push_back( matches[i]); }
        
        // 3. Find Essential Matrix & Recover Pose
        vector<Point2f> p1, p2;
        for( int i = 0; i < good_matches.size(); i++ ) {
            p1.push_back( k1[good_matches[i].queryIdx].pt );
            p2.push_back( k2[good_matches[i].trainIdx].pt );
        }
        
        if (p1.size() < 5) continue; // Safety check

        Mat E, R, t, mask;
        E = findEssentialMat(p1, p2, focal, pp, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, p1, p2, R, t, focal, pp, mask);


        // ... (After recoverPose) ...

        // 1. Create Projection Matrices for Triangulation
        // P1 = Identity (Origin)
        // P2 = R|t (The new pose we just calculated)
        Mat P1 = Mat::eye(3, 4, CV_64F); // 3x4 projection matrix
        Mat P2 = Mat::zeros(3, 4, CV_64F);
        
        R.copyTo(P2.rowRange(0,3).colRange(0,3));
        t.copyTo(P2.rowRange(0,3).col(3));

        // 2. Triangulate
        // We need to convert Point2f to Point2d for the function
        Mat points4D;
        triangulatePoints(P1, P2, p1, p2, points4D);
        
        // 3. Transform points to World Coordinate System
        // (The points are currently relative to Camera 1. We need to apply the global R_f and t_f)
        vector<Point3f> localPoints, worldPoints;
        points4D.convertTo(points4D, CV_32F); // Convert to float for our helper
        mat2Points(points4D, localPoints);
        
        for(Point3f pt : localPoints) {
            // Apply global rotation/translation to the point
            // X_world = R_final * X_local + t_final
            
            // Note: This is rough linear algebra for a simple visual
            Mat ptMat = (Mat_<double>(3,1) << pt.x, pt.y, pt.z);
            Mat worldPtMat = R_f * ptMat + t_f;
            
            worldPoints.push_back(Point3f(worldPtMat.at<double>(0), worldPtMat.at<double>(1), worldPtMat.at<double>(2)));
        }

        // 4. Save to file
        savePoints("map.xyz", worldPoints);
        
        // --- TRAJECTORY UPDATE ---
        
        // We need to invert the logic because t is negative-z
        // Standard formula: t_final = t_final + (Scale * (R_final * t))
        // Note: We use 't' directly here. If it goes backward, we flip the sign of t.
        
        double scale = getAbsoluteScale(numFrame)/1000; // Replace with getAbsoluteScale(numFrame) if you want accuracy
        
        t_f = t_f + scale * (R_f * t);
        R_f = R * R_f;

        // --- DRAWING ---
        
        // Convert x,z coordinates to map coordinates (center at 300,300)
        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 500; // Using Z as Y for 2D plotting
        
        circle(traj, Point(x, y), 1, Scalar(0, 255, 0), 2);
        
        imshow("Road Facing Camera", img_2);
        imshow("Trajectory", traj);
        waitKey(1);
    }

    return 0;
}