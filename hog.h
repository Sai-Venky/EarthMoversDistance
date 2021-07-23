#include <iostream>
#include <opencv2/opencv.hpp>
#include<array>
#include <math.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <dirent.h>
using namespace std;


class HOG {
     cv::Mat image;
     int cell_size, bin_size, angle_unit;
     double max_mag; 
     int min_angle, max_angle;
     double mod;

 public:
     HOG(cv::Mat img, int cell, int bin) {
        image = img;
        double minVal; 
        double maxVal; 
        cv::Point minLoc; 
        cv::Point maxLoc;
        minMaxLoc( image, &minVal, &maxVal, &minLoc, &maxLoc );
        image.convertTo(image, CV_32F, 1.0 /maxVal, 0);
        cv::sqrt(image,image);
        image = image * 255;
        cell_size = cell;
        bin_size = bin;
        angle_unit = 360 / bin_size;
        max_mag = 0;
     }
    
    cv::Mat extract() {
        int height=image.size().height;
        int width = image.size().width;


        cv::Mat gradient_values_x,abs_grad_x,abs_grad_y,gradient_values_y,gradient_magnitude,gradient_angle, cell_magnitude, cell_angle;
        cv::Sobel(image, gradient_values_x,CV_64F, 1, 0,5);
        cv::Sobel(image,gradient_values_y ,CV_64F, 0, 1,5);
        cv::convertScaleAbs(gradient_values_x, abs_grad_x);
        cv::convertScaleAbs(gradient_values_y, abs_grad_y);
        cv::addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0,gradient_magnitude);
        cv::phase(gradient_values_x, gradient_values_y, gradient_angle ,true);

        gradient_magnitude = abs(gradient_magnitude);

        int h = int(image.size().height/cell_size);
        int w = int(image.size().width/cell_size);
        vector<vector<vector<double>>> cell_gradient_vector(h,vector<vector<double>>(w ,vector<double>(bin_size,0)));


        for(auto i =0;i<cell_gradient_vector.size();i++) {
            for(auto j=0;j<cell_gradient_vector[i].size();j++) {
                cell_magnitude = gradient_magnitude(cv::Range(i * cell_size,(i + 1) * cell_size), cv::Range(j * cell_size,(j + 1) * cell_size));
                cell_angle = gradient_angle(cv::Range(i*cell_size,(i + 1)*cell_size), cv::Range(j * cell_size,(j + 1)*cell_size));
                cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle);
            }
        }
        
 
        cv::Mat image_output = cv::Mat(height,width, CV_64F, double(0));;
        image_output = render_gradient(image_output, cell_gradient_vector);
        return image_output;

    }
    
    
    vector<double> cell_gradient(cv::Mat &cell_magnitude, cv::Mat &cell_angle) {
        vector<double> orientation_centers(bin_size, 0);
        for(int i =0;i<cell_magnitude.size().height;i++) {
            for(int j=0;j<cell_magnitude.size().width;j++) {
                double gradient_strength = cell_magnitude.at<double>(i,j);
                double gradient_angle = cell_angle.at<double>(i,j);
                get_closest_bins(gradient_angle);
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / angle_unit)));
                orientation_centers[max_angle] += (gradient_strength * (mod / angle_unit));
            }
        }
        
        if(max_mag < orientation_centers[min_angle]) max_mag = orientation_centers[min_angle];
        if(max_mag < orientation_centers[max_angle]) max_mag = orientation_centers[max_angle];
        return orientation_centers;
    }
    

    void get_closest_bins(int gradient_angle) {
        int idx = int(gradient_angle / angle_unit);
        double mod_val = gradient_angle % angle_unit;
        if(idx == bin_size) {
            min_angle = idx - 1;
            max_angle = idx % bin_size;
            mod = mod_val;
        } else {
            min_angle = idx;
            max_angle = (idx+1) % bin_size;
            mod = mod_val;
        }
    }
    
    cv::Mat render_gradient(cv::Mat &image_output,vector<vector<vector<double>>> & cell_gradient_vector) {
        int cell_width=cell_size/2;
        double max_mag_var = max_mag;
        for(auto i =0;i<cell_gradient_vector.size();i++)
        {
            for(auto j=0;j<cell_gradient_vector[i].size();j++)
            {
                vector<double> cell_grad = cell_gradient_vector[i][j];
                for(auto k=0;k<cell_grad.size();k++) {
                   cell_grad[k] /= max_mag_var;
                }
                int angle = 0;
                int angle_gap = angle_unit;
                double pi = 3.14159265359;
                for(auto k=0;k<cell_grad.size();k++)
                {
                    double magnitude=cell_grad[k];
                    float angle_radian = angle* (pi/180); 
                    int x1 = int(i * cell_size + magnitude * cell_width *cos(angle_radian));
                    int y1 = int(j * cell_size + magnitude * cell_width * sin(angle_radian));
                    int x2 = int(i* cell_size - magnitude * cell_width *cos(angle_radian));
                    int y2 = int(j * cell_size - magnitude * cell_width * sin(angle_radian));
                    cv::line(image_output, cv::Point(y1, x1), cv::Point(y2, x2), int(255 * sqrt(magnitude)));
                    angle += angle_gap;
                }
            }
        }
        return image_output;
    }
    
};

int main()
{       
  Load the image and template in _image_val and _template_val     
  cvtColor(_image_val, _image_val, cv::COLOR_BGR2GRAY);
  cvtColor(_template_val, _template_val, cv::COLOR_BGR2GRAY);

  HOG* hogImg = new HOG(_image_val, 2, 8);
  HOG* hogTemplate = new HOG(_template_val, 2, 8);

  cv::Mat _image = hogImg->extract();
  cv::Mat _template = hogTemplate->extract();
}
