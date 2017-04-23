#ifndef XETHRU_MAP2D_UTILS_H
#define XETHRU_MAP2D_UTILS_H

#include <fstream>

#include "boost/date_time/posix_time/posix_time.hpp" //include all types plus i/o
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/exception/all.hpp>
#include <boost/make_shared.hpp>
#include <boost/throw_exception.hpp>

#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/regex.h>
#include <boost/regex.hpp>

namespace fs = boost::filesystem;
//namespace po = boost::program_options;

/// Escape all regex special chars
inline void escapeRegex(std::string& regex) {
    boost::replace_all(regex, "\\", "\\\\");
    boost::replace_all(regex, "^", "\\^");
    boost::replace_all(regex, ".", "\\.");
    boost::replace_all(regex, "$", "\\$");
    boost::replace_all(regex, "|", "\\|");
    boost::replace_all(regex, "(", "\\(");
    boost::replace_all(regex, ")", "\\)");
    boost::replace_all(regex, "[", "\\[");
    boost::replace_all(regex, "]", "\\]");
    boost::replace_all(regex, "*", "\\*");
    boost::replace_all(regex, "+", "\\+");
    boost::replace_all(regex, "?", "\\?");
    boost::replace_all(regex, "/", "\\/");
}

inline std::string wildcardsToRegex(std::string const& aStrWithWildcards) {
    //: Escape all regex special chars
    std::string regexStr = aStrWithWildcards;
    escapeRegex(regexStr);
    //: Convert chars '*?' to their regex equivalents:
    boost::replace_all(regexStr, "\\?", ".");
    boost::replace_all(regexStr, "\\*", ".*");
    return regexStr;
}

inline void appendFilePaths(std::vector<std::string>& aResult, fs::path const& aPath, std::string const& mask) {
    boost::regex regexQuery(wildcardsToRegex(mask), boost::regex::icase);
    fs::recursive_directory_iterator endItr;
    for (fs::recursive_directory_iterator fi(aPath); fi != endItr; ++fi) {
        //: Skip if not a file
        if (!fs::is_regular_file(fi->status()))
            continue;
        //: Skip if no match
        if (!boost::regex_match(fi->path().filename().string(), regexQuery))
            continue;
        aResult.push_back(fi->path().string());
    }
}

inline std::vector<std::string> split(const std::string & str, const std::string & sep) {
    std::vector<std::string> result;
    if (str.empty())
        return result;

    std::string::size_type CurPos(0), LastPos(0);
    while (1) {
        CurPos = str.find(sep, LastPos);
        if (CurPos == std::string::npos)
            break;

        std::string sub =
            str.substr(LastPos,
                std::string::size_type(CurPos -
                    LastPos));
        if (sub.size())
            result.push_back(sub);

        LastPos = CurPos + sep.size();
    }

    std::string sub = str.substr(LastPos);
    if (sub.size())
        result.push_back(sub);

    return result;
}

inline std::vector<std::string> findFilesPaths(fs::path const& path, std::string const& masks_list) {
    std::vector<std::string> result;
    std::vector<std::string> inputMasksList = split(masks_list, ";");
    BOOST_FOREACH(std::string input_mask, inputMasksList) {
        appendFilePaths(result, path, input_mask);
    }
    return result;
}

struct Utils {
    enum Color {
        Black = 0,
        White = 1,
        Grey = 2,
        Red = 3,
        Green = 4,
        Blue = 5,
        Yellow = 6,
        Cyan = 7,
        Magenta = 8,
        DarkRed = 9,
        DarkGreen = 10,
        DarkBlue = 11,
        DarkYellow = 12,
        DarkCyan = 13,
        DarkMagenta = 14,
        DarkGrey = 15,
        NumOfColors
    };

    static cv::Scalar getColor(Color c) {
        static const cv::Scalar g_colors[NumOfColors] = {
            cv::Scalar(  0,  0,  0), //Black,
            cv::Scalar(255,255,255), //White,
            cv::Scalar(128,128,128), //Grey,
            cv::Scalar(  0,  0,255), //Red,
            cv::Scalar(  0,255,  0), //Green,
            cv::Scalar(255,  0,  0), //Blue,
            cv::Scalar(  0,255,255), //Yellow,
            cv::Scalar(255,255,  0), //Cyan,
            cv::Scalar(255,  0,255), //Magenta,
            cv::Scalar(  0,  0,128), //DarkRed,
            cv::Scalar(  0,128,  0), //DarkGreen,
            cv::Scalar(128,  0,  0), //DarkBlue,
            cv::Scalar(  0,128,128), //DarkYellow,
            cv::Scalar(128,128,  0), //DarkCyan,
            cv::Scalar(128,  0,128), //DarkMagenta,
            cv::Scalar( 64, 64, 64), //DarkGrey,
        };
        if((int)c < 0 || (int)c >= NumOfColors)
            return cv::Scalar(0,0,128); //: Dark Red to indicate error value
        return g_colors[c];
    }


    template<typename t_CoordType>
    static cv::Rect_<t_CoordType> inflate(cv::Rect_<t_CoordType> const& r, t_CoordType delta) {
        return cv::Rect_<t_CoordType>(r.x - delta, r.y - delta, r.width + delta, r.height + delta);
    }

    static cv::Mat getSaturationChannel(cv::Mat const& src_bgr_image) {
        cv::Mat hsv_image = src_bgr_image.clone();
        cv::cvtColor(hsv_image, hsv_image, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv_image, hsv_channels);
        return hsv_channels[1];
    }

    static cv::Mat makeObjectMask(cv::Mat const& src_bgr_image, int saturation_threshold, int brightness_threshold) {
        cv::Mat hsv_image = src_bgr_image.clone();
        cv::cvtColor(hsv_image, hsv_image, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv_image, hsv_channels);
//        cv::imshow("image", src_bgr_image);
//        cv::waitKey();
//        cv::imshow("saturation", hsv_channels[1]);
//        cv::waitKey();

//        cv::Mat bright_mask = (hsv_channels[2] >= brightness_threshold)(cv::Rect(1, 1, src_bgr_image.cols - 2, src_bgr_image.rows - 2));
//        cv::Mat diff_image = singleChannelToMaxDiff(hsv_channels[1]);
//        for(int diff_threshold = 8; diff_threshold < 60; diff_threshold += 4) {
//            std::stringstream strstr;
//            strstr << "saturation diff with diff_threshold = " << diff_threshold;
//            std::string winname = strstr.str();
//            cv::imshow(winname, (diff_image < diff_threshold) & bright_mask);
//            cv::waitKey();
//            cv::destroyWindow(winname);
//        }
        return (hsv_channels[1] < saturation_threshold) & (hsv_channels[2] >= brightness_threshold);
    }

    static cv::Mat singleChannelToMaxDiff(cv::Mat const& image) {
        cv::Mat result = singleChannelToMaxDiffAlongNorthEast(image); //: 'NortEast' and 'SouthWest' neighbors
        result = cv::max(result, singleChannelToMaxDiffAlongShift(image, 1, 0)); //: 'East' and 'West' neighbors
        result = cv::max(result, singleChannelToMaxDiffAlongShift(image, 1, 1)); //: 'SouthEast' and 'NorthWest' neighbors
        result = cv::max(result, singleChannelToMaxDiffAlongShift(image, 0, 1)); //: 'South' and 'North' neighbors
        return result;
    }


    static cv::Mat makeHorizontallyStitchedImage(cv::Mat const& left_image, cv::Mat const& right_image) {
        int rows = std::max(left_image.size().height, right_image.size().height);
        int cols = left_image.size().width + 1 + right_image.size().width;
        cv::Mat result = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0));
        left_image.copyTo(result.colRange(0, left_image.size().width).rowRange(0, left_image.size().height));
        right_image.copyTo(result.colRange(left_image.size().width + 1, cols).rowRange(0, right_image.size().height));
        return result;
    }

    static cv::Mat makeVerticallyStitchedImage(cv::Mat const& top_image, cv::Mat const& bottom_image) {
        int rows = top_image.size().height + 1 + bottom_image.size().height;
        int cols = std::max(top_image.size().width, bottom_image.size().width);
        cv::Mat result = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0));
        top_image.copyTo(result.colRange(0, top_image.size().width).rowRange(0, top_image.size().height));
        bottom_image.copyTo(result.colRange(0, bottom_image.size().width).rowRange(top_image.size().height + 1, rows));
        return result;
    }

//private:
    static cv::Mat singleChannelToMaxDiff(cv::Mat const& image1, cv::Mat const& image2) {
        return cv::max(image1, image2) - cv::min(image1, image2);
    }

    static cv::Mat singleChannelToShiftDiff(cv::Mat const& image, int shift_x, int shift_y) {
        cv::Mat image0(image, cv::Rect(0, 0, image.cols - 1, image.rows - 1));
        cv::Mat image_shifted(image, cv::Rect(shift_x, shift_y, image.cols - 1, image.rows - 1));
        return singleChannelToMaxDiff(image0, image_shifted);
    }
    static cv::Mat singleChannelToShiftMin(cv::Mat const& image, int shift_x, int shift_y) {
        cv::Mat image0(image, cv::Rect(0, 0, image.cols - 1, image.rows - 1));
        cv::Mat image_shifted(image, cv::Rect(shift_x, shift_y, image.cols - 1, image.rows - 1));
        return cv::min(image0, image_shifted);
    }
    static cv::Mat shiftDiffToMaxDiffAlongShift(cv::Mat const& shift_diff, int shift_x, int shift_y) {
        cv::Mat shift_diff0(shift_diff, cv::Rect(0, 0, shift_diff.cols - 1, shift_diff.rows - 1));
        cv::Mat shifted_back(shift_diff, cv::Rect(shift_x, shift_y, shift_diff.cols - 1, shift_diff.rows - 1));
        return cv::max(shift_diff0, shifted_back);
    }
    static cv::Mat singleChannelToMaxDiffAlongShift(cv::Mat const& image, int shift_x, int shift_y) {
        return shiftDiffToMaxDiffAlongShift(singleChannelToShiftDiff(image, shift_x, shift_y), shift_x, shift_y);
    }

    //: ok, let the case shift_x = 1, shift_y = -1 be a special case. let's call it 'NorthEast' shift.
    static cv::Mat shiftDiffToMaxDiffAlongNorthEast(cv::Mat const& shift_diff) {
        cv::Mat shift_diff0(shift_diff, cv::Rect(0, 1, shift_diff.cols - 1, shift_diff.rows - 1));
        cv::Mat shifted_back(shift_diff, cv::Rect(1, 0, shift_diff.cols - 1, shift_diff.rows - 1));
        return cv::max(shift_diff0, shifted_back);
    }
    static cv::Mat singleChannelToMaxDiffAlongNorthEast(cv::Mat const& image) {
        cv::Mat image0(image, cv::Rect(0, 1, image.cols - 1, image.rows - 1));
        cv::Mat image_shifted(image, cv::Rect(1, 0, image.cols - 1, image.rows - 1));
        return shiftDiffToMaxDiffAlongNorthEast(singleChannelToMaxDiff(image0, image_shifted));
    }
};

template<typename t_Number>
inline t_Number sq(t_Number v) { return v * v; }

#endif