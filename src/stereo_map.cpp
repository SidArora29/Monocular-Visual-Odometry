#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp> // VITAL: Includes Stereo functions
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// KITTI Calibration Data (Sequence 00)
const double focal_length = 718.8560;
const double baseline = 0.537; // Meters between Left and Right camera
const double cx = 607.1928;
const double cy = 185.2157;

int main() {
    // 1. Load ONE pair of images
    // UPDATE THESE PATHS to your actual files
    Mat img_left = imread("/media/sid/PortableSSD/data_odometry_gray/dataset/sequences/00/image_0/000000.png", IMREAD_GRAYSCALE);
    Mat img_right = imread("/media/sid/PortableSSD/data_odometry_gray/dataset/sequences/00/image_1/000000.png", IMREAD_GRAYSCALE);

    if (img_left.empty() || img_right.empty()) {
        cout << "Error: Could not load images. Check paths!" << endl;
        return -1;
    }

    // 2. Setup Stereo Matcher (SGBM)
    // These parameters are tuned for the KITTI dataset
    int min_disp = 0;
    int num_disp = 16 * 6; // 96
    int block_size = 5;

    Ptr<StereoSGBM> stereo = StereoSGBM::create(
        min_disp,
        num_disp,
        block_size,
        8 * 3 * block_size * block_size,  // P1
        32 * 3 * block_size * block_size, // P2
        1,  // disp12MaxDiff
        0,  // preFilterCap
        10, // uniquenessRatio
        100,// speckleWindowSize
        32  // speckleRange
    );

    cout << "Computing disparity..." << endl;
    Mat disparity;
    stereo->compute(img_left, img_right, disparity);

    // 3. Convert Disparity to 3D Points
    // Formula: Z = (f * B) / d
    vector<Point3f> objectPoints;
    
    for (int v = 0; v < disparity.rows; v++) {
        for (int u = 0; u < disparity.cols; u++) {
            // SGBM outputs fixed-point values multiplied by 16. 
            // We must divide by 16.0 to get real pixels.
            float d = disparity.at<short>(v, u) / 16.0;

            // Filter out bad matches (disparity <= 0 means infinity or error)
            if (d <= 0.0 || d >= num_disp - 1) continue;

            // Triangulate
            float z = (focal_length * baseline) / d;
            float x = (u - cx) * z / focal_length;
            float y = (v - cy) * z / focal_length;

            // Filter depth (Only keep points 0.5m to 50m away)
            if (z > 0.5 && z < 50.0) {
                // Invert Y because computer graphics Y is up, OpenCV Y is down
                objectPoints.push_back(Point3f(x, -y, z));
            }
        }
    }

    // 4. Save to stereo.xyz
    cout << "Saving " << objectPoints.size() << " points to stereo.xyz..." << endl;
    ofstream f("stereo.xyz");
    for (auto p : objectPoints) {
        f << p.x << " " << p.y << " " << p.z << endl;
    }
    f.close();

    cout << "Done! Drag stereo.xyz to the viewer." << endl;
    return 0;
}