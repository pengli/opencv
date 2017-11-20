#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

static void help()
{
    std::cout << "Usage: visual_style <camera_id>" << std::endl;
}

int main(int argc, char **argv)
{
    VideoCapture cap;
    int fd = 0;
    //const string model = "fast_neural_style_instance_norm_feathers.t7";
    //const string model = "candy.t7";
    const string model = "the_scream.t7";

    if (argc > 2)
    {
        help();
        return -1;
    }
    else if (argc == 2)
    {
        fd = atoi(argv[1]);
    }

    Net net = dnn::readNetFromTorch(model);
    if (net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "torch model: " << model << std::endl;
        return -1;
    }

    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_OPENCL);

    cap.open(fd);
    if (!cap.isOpened())
    {
        std::cerr << "can't open camera " << fd << std::endl;
        return -1;
    }

    Mat img;
    cap >> img;
    Mat inputBlob = blobFromImage(img, 1.0, Size(), Scalar(103.939, 116.779, 123.68), false);

    net.setInput(inputBlob);
    Mat out = net.forward();

    char ret;
    Mat out_img;
    UMat frame;
    int i = 0;
    cv::TickMeter t;
    while (i++ < 50)
    {
        cap >> img;
        if (img.empty())
        {
            std::cerr << "cannot find frame" << std::endl;
            break;
        }

        inputBlob = blobFromImage(img, 1.0, Size(), Scalar(103.939, 116.779, 123.68), false);

        net.setInput(inputBlob);
        t.start();
        out = net.forward();
        t.stop();

        // Deprocessing.
        getPlane(out, 0, 0) += 103.939;
        getPlane(out, 0, 1) += 116.779;
        getPlane(out, 0, 2) += 123.68;
        out = cv::min(cv::max(0, out), 255);

        out.convertTo(out_img, CV_8U);
        std::vector<Mat> plane(3);
        plane[0] = getPlane(out_img, 0, 0);
        plane[1] = getPlane(out_img, 0, 1);
        plane[2] = getPlane(out_img, 0, 2);
        merge(plane, frame);

        imshow("video", frame);
        if ((ret = waitKey(20)) > 0) break;
    }
    std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << std::endl;
    return 0;
}

