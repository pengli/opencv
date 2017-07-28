#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::dnn;

static void help()
{
    std::cout << "Usage: age_gender <camera_id>" << std::endl;
}

int main(int argc, char **argv)
{
    VideoCapture cap;
    int fd = 1;
    UMat frame, small, gray;
    CascadeClassifier cascade;
    string cascadeName = "../../data/haarcascades/haarcascade_frontalface_alt.xml";
    String gender_modelTxt = "deploy_gender.prototxt";
    String gender_modelBin = "gender_net.caffemodel";
    String age_modelTxt = "deploy_age.prototxt";
    String age_modelBin = "age_net.caffemodel";

    if (argc > 2)
    {
        help();
        return -1;
    }
    else if (argc == 2)
    {
        fd = atoi(argv[1]);
    }

    Net gender_net = dnn::readNetFromCaffe(gender_modelTxt, gender_modelBin);
    Net age_net = dnn::readNetFromCaffe(age_modelTxt, age_modelBin);

    if (gender_net.empty() || age_net.empty())
    {
        std::cerr << "Can't load network by using the following files: " << std::endl;
        std::cerr << "prototxt:   " << gender_modelTxt << age_modelTxt << std::endl;
        std::cerr << "caffemodel: " << gender_modelBin << age_modelBin << std::endl;
        return -1;
    }

    if (!cascade.load(cascadeName))
    {
        std::cerr << "can't load classifier cascade" << std::endl;
        return -1;
    }

    cap.open(fd);
    if (!cap.isOpened())
    {
        std::cerr << "can't open camera " << fd << std::endl;
        return -1;
    }

    char ret;
    float fx = 0.5;
    int M = 227;
    String ages[] = {"0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60-"};

    vector<Rect> faces;
    UMat im, f_im, input, crop;

    while(true)
    {
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "cannot find frame" << std::endl;
            break;
        }

        resize(frame, small, Size(), fx, fx, INTER_LINEAR);

        cvtColor(small, gray, COLOR_BGR2GRAY);

        cascade.detectMultiScale(gray, faces);

        for (size_t i = 0; i < faces.size(); i++)
        {
            Rect f = faces[i];
            int x = f.x / fx;
            int y = f.y / fx;
            int w = f.width / fx;
            int h = f.height / fx;

            getRectSubPix(frame, Size(w, h), Point2f(x + w / 2, y + h / 2), crop);

            if (!crop.empty())
            {
                resize(crop, im, Size(256, 256));

                cvtColor(im, im, cv::COLOR_BGR2RGB);

                im.convertTo(f_im, CV_32FC3);

                resize(f_im, input, Size(M, M));

                Mat inputBlob = blobFromImage(input.getMat(ACCESS_READ));
                gender_net.setInput(inputBlob, "data");
                age_net.setInput(inputBlob, "data");

                Mat gender_prob = gender_net.forward("prob");
                Mat age_prob = age_net.forward("prob");

                float *ptr = gender_prob.ptr<float>(0);
                String gender = (ptr[0] > ptr[1]) ? "M" : "F";

                double maxVal;
                Point maxNumber;
                minMaxLoc(age_prob, NULL, &maxVal, NULL, &maxNumber);
                String age = ages[maxNumber.x];

                rectangle(frame, Point(x, y + 3), Point(x + w, y + h), Scalar(30, 255, 30), 2);
                putText(frame, gender + ":" + age,  Point(x, y), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 0), 2);
            }
        }

        imshow("video", frame);

        if ((ret = waitKey(20)) > 0) break;
    }
}

