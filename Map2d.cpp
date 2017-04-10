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

class Map2d {
public:
    double map_min_x_in_m_;
    double map_max_x_in_m_;
    double map_min_y_in_m_;
    double map_max_y_in_m_;
    double pixels_in_one_m_;

    Map2d() {
        map_min_x_in_m_ = -4;
        map_min_y_in_m_ = -5;
        map_max_x_in_m_ =  4;
        map_max_y_in_m_ =  5;
        pixels_in_one_m_ = 50;
    }

    int getWidhthInPixels() const { return (int)(pixels_in_one_m_ * (map_max_x_in_m_ - map_min_x_in_m_)); }
    int getHeightInPixels() const { return (int)(pixels_in_one_m_ * (map_max_y_in_m_ - map_min_y_in_m_)); }

    cv::Size getSizeInPixels() const {
        return cv::Size(getWidhthInPixels(), getHeightInPixels());
    }

    template<typename t_cvPointLike>
    cv::Point2i toPixels(t_cvPointLike const& map_2d_coords) const {
        return cv::Point2i((map_2d_coords.x - map_min_x_in_m_) * pixels_in_one_m_
            , getHeightInPixels() - (map_2d_coords.y - map_min_y_in_m_) * pixels_in_one_m_);
    }

    template<typename t_cvPointLike>
    cv::Point2f fromPixels(t_cvPointLike const& pixel_2d_coords) const {
        return cv::Point2f(pixel_2d_coords / pixels_in_one_m_ + map_min_x_in_m_
            , getHeightInPixels() - (pixel_2d_coords.y / pixels_in_one_m_ + map_min_y_in_m_));
    }
};

template<typename t_Distance, typename t_cvPointLike, typename t_PixelDistanceFunctor>
inline void makePointDistanceMatrix(cv::Mat_<t_Distance>& result, t_cvPointLike const& pixel_2d_coords, t_PixelDistanceFunctor const& pixel_distance_functor) {
    cv::Size size = result.size();
    for (int row = 0; row < size.height; ++row) {
        t_Distance* p = result[row];
        double dy_sq = sq(row - pixel_2d_coords.y);
        for (int col = 0; col < size.width; ++col) {
            double distance_in_pixels = std::sqrt(dy_sq + sq(col - pixel_2d_coords.x));
            p[col] = (t_Distance)pixel_distance_functor(distance_in_pixels);
        }
    }
}

struct RadarRawMeas {
    struct Params {
        double first_bin_offset_in_m_;
        double bin_size_in_m_;
        double max_amplitude_value_;

        Params()
            : first_bin_offset_in_m_(0.0514) //fixme?
            , bin_size_in_m_(0.0514) //fixme?
            , max_amplitude_value_(0.01) //fixme?
        {}
    };

    struct Bin {
        double amplitude_;

        Bin(double amplitude = 0) : amplitude_(amplitude) {}
    };

    RadarRawMeas() : radar_index_(0) {}

    int radar_index_;
    std::vector<Bin> bins_;
};

typedef std::vector<RadarRawMeas> RadarRawMeasSet;

class RadarRawMeasSetStreamBase {
public:
    virtual ~RadarRawMeasSetStreamBase() {}
    virtual bool moveNext() = 0;
    virtual RadarRawMeasSet const& getCurrent() const = 0;
};



template<int c_numOfRadars>
class RadarsMap2dT {
public:
    static const int c_NumOfRadars = c_numOfRadars;

    struct Params {
        Map2d workarea_map_;
        cv::Point2f radar_pos_in_m_[c_NumOfRadars];
        RadarRawMeas::Params radar_raw_frame_params_;

        Params() {}

    };

    RadarsMap2dT(Params const& params) : params_(params), pix2bins_converter_(params) {
        setParams(params);
    }

    Params const& getParams() const { return params_; }
    void setParams(Params const& params) {
        params_ = params;
        pix2bins_converter_ = Pixels2BinsDistanceConverter(params_);
        makeMats(params_.workarea_map_.getSizeInPixels());
        for (int i = 0; i < c_NumOfRadars; ++i)
            makeRadarDistanceMap(precalculated_radar_distance_maps_in_bins_[i], params_.radar_pos_in_m_[i]);
    }

    cv::Mat getColorMap() const { return color_map_; }
    cv::Mat_<uchar> getHeatMap() const { return heat_map_; }

    void plotMeas(RadarRawMeasSet const& meas) {
        for (std::size_t i = 0; i < meas.size(); ++i)
            plotMeas(meas[i]);
        //fixme: make better heat_map_
        color_map_channels_[2] = heat_map_ = (color_map_channels_[0] & color_map_channels_[1]);
        cv::merge(color_map_channels_, sizeof(color_map_channels_) / sizeof(color_map_channels_[0]), color_map_);
    }

private:
    void makeMats(cv::Size map_size_in_pixels) {
        for (int i = 0; i < c_NumOfRadars; ++i) {
            precalculated_radar_distance_maps_in_bins_[i].create(map_size_in_pixels);
            color_map_channels_[i].create(map_size_in_pixels);
        }
        color_map_.create(map_size_in_pixels);
        heat_map_.create(map_size_in_pixels);
    }

    void makeRadarDistanceMap(cv::Mat_<uchar>& result_radar_distance_maps_in_bins, cv::Point2f const& radar_pos_in_m) const {
        cv::Point radar_pos_in_pix = params_.workarea_map_.toPixels(radar_pos_in_m);
        makePointDistanceMatrix(result_radar_distance_maps_in_bins, radar_pos_in_pix, pix2bins_converter_);
    }

    void plotMeas(RadarRawMeas const& radar_raw_meas) {
        if (radar_raw_meas.radar_index_ < 0 || radar_raw_meas.radar_index_ >= c_NumOfRadars) {
            std::cout << "Warning: got radar_index = " << radar_raw_meas.radar_index_ << std::endl;
            //: do nothing:
            return;
        }
        //fixme?
        std::vector<uchar> amplitudes_as_uchars(256, 0);
        double amplitude_scale_to_uchar = 255. / params_.radar_raw_frame_params_.max_amplitude_value_;
        for (std::size_t i = 0; i < radar_raw_meas.bins_.size(); ++i) {
            if (radar_raw_meas.bins_[i].amplitude_ < 0 || radar_raw_meas.bins_[i].amplitude_ > params_.radar_raw_frame_params_.max_amplitude_value_) {
                std::cout << "Warning: got amplitude value = " << radar_raw_meas.bins_[i].amplitude_ << std::endl;
                amplitudes_as_uchars[i] = 255;
            } else
                amplitudes_as_uchars[i] = (uchar)(radar_raw_meas.bins_[i].amplitude_ * amplitude_scale_to_uchar);
        }
        cv::LUT(precalculated_radar_distance_maps_in_bins_[radar_raw_meas.radar_index_], amplitudes_as_uchars, color_map_channels_[radar_raw_meas.radar_index_]);
        //cv::imshow(radar_raw_meas.radar_index_ ? "radar_distance_maps_in_bins-1" : "radar_distance_maps_in_bins-0", precalculated_radar_distance_maps_in_bins_[radar_raw_meas.radar_index_]);
    }

private:
    struct Pixels2BinsDistanceConverter {
        double offset_in_pixels_;
        double bins_in_one_pixel_;

        Pixels2BinsDistanceConverter(Params const& params) {
            offset_in_pixels_ = params.radar_raw_frame_params_.first_bin_offset_in_m_ * params.workarea_map_.pixels_in_one_m_;
            bins_in_one_pixel_ = 1. / (params.workarea_map_.pixels_in_one_m_ * params.radar_raw_frame_params_.bin_size_in_m_);
        }

        double operator()(double distance_in_pixels) const {
            return (distance_in_pixels - offset_in_pixels_) * bins_in_one_pixel_;
        }
    };

    Params params_;

    Pixels2BinsDistanceConverter pix2bins_converter_;

    cv::Mat_<uchar> precalculated_radar_distance_maps_in_bins_[c_NumOfRadars];
    cv::Mat_<uchar> color_map_channels_[3];
    cv::Mat_<cv::Vec3b> color_map_;
    cv::Mat_<uchar> heat_map_;
};

typedef RadarsMap2dT<2> TwoRadarsMap2d;
typedef RadarsMap2dT<3> ThreeRadarsMap2d;


class TwoAmplCsvFilesStream : public RadarRawMeasSetStreamBase {
public:
    TwoAmplCsvFilesStream(std::string const& radar0_amp_file_path, std::string const& radar1_amp_file_path)
        : radar0_amp_file_(radar0_amp_file_path)
        , radar1_amp_file_(radar1_amp_file_path)
        , current_meas_set_(2)
    {
        current_meas_set_[0].radar_index_ = 0;
        current_meas_set_[1].radar_index_ = 1;
    }

    virtual bool moveNext() {
        bool ok = true;
        ok = ok && radar0_amp_file_.good();
        ok = ok && radar1_amp_file_.good();
        std::string csv_line0;
        ok = ok && std::getline(radar0_amp_file_, csv_line0);
        std::string csv_line1;
        ok = ok && std::getline(radar1_amp_file_, csv_line1);
        if (ok) {
            readCsvLine(current_meas_set_[0], csv_line0, 1);
            readCsvLine(current_meas_set_[1], csv_line1, 1);
        }
        return ok;
    }

    virtual RadarRawMeasSet const& getCurrent() const { return current_meas_set_; }

private:
    static void readCsvLine(RadarRawMeas& res, std::string const& csv_line, int start_field_index) {
        std::size_t field_start_pos = 0;
        std::size_t field_end_pos = 0;
        std::size_t field_index = 0;

        //: skip first start_field_index fields:
        while (field_index < start_field_index && (field_end_pos = csv_line.find(',', field_end_pos + 1)) != std::string::npos)
            ++field_index;
        if (field_end_pos == std::string::npos) {
            //: empty list of bins:
            res.bins_.resize(0);
            return;
        }

        field_index = 0;
        do {
            field_start_pos = field_end_pos + 1;
            field_end_pos = csv_line.find(',', field_start_pos);
            std::stringstream strstr;
            if(field_end_pos != std::string::npos)
                strstr << csv_line.substr(field_start_pos, field_end_pos - field_start_pos);
            else
                strstr << csv_line.substr(field_start_pos);
            if (res.bins_.size() <= field_index)
                res.bins_.resize(field_index + 1);
            strstr >> res.bins_[field_index].amplitude_;
            ++field_index;
        } while (field_end_pos != std::string::npos);
        //: cut old elements after field_index-th:
        res.bins_.resize(field_index + 1);
    }

private:
    std::ifstream radar0_amp_file_;
    std::ifstream radar1_amp_file_;
    RadarRawMeasSet current_meas_set_;
};

void printHelp(char const* program_name) {
    std::cout << "Usage:\n"
        << program_name << " path_to_radar0_amplitudes.csv, path_to_radar1_amplitudes.csv" << std::endl;
}

void printParams(char const* path_to_radar0_amplitudes, char const* path_to_radar1_amplitudes, TwoRadarsMap2d::Params const& params) {
    std::cout << "Starting with following parameters:\n"
        << "Files:\n"
        << "    " << path_to_radar0_amplitudes << "\n"
        << "    " << path_to_radar1_amplitudes << "\n"
        << "Workarea Map:"
        << "    X: [" << params.workarea_map_.map_min_x_in_m_ << ", " << params.workarea_map_.map_max_x_in_m_ << "]\n"
        << "    Y: [" << params.workarea_map_.map_min_y_in_m_ << ", " << params.workarea_map_.map_max_y_in_m_ << "]\n"
        << "    prixel size: " << 1 / params.workarea_map_.pixels_in_one_m_ << "\n"
        << "Radars positions:"
        << "    " << params.radar_pos_in_m_[0] << "\n"
        << "    " << params.radar_pos_in_m_[1] << "\n"
        << "Radars parameters:"
        << "    bin_size_in_m: " << params.radar_raw_frame_params_.bin_size_in_m_ << "\n"
        << "    first_bin_offset_in_m: " << params.radar_raw_frame_params_.first_bin_offset_in_m_ << "\n"
        << "    max_amplitude_value: " << params.radar_raw_frame_params_.max_amplitude_value_ << "\n"
        << std::endl;
}


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Error: Incorrect number of arguments: " << (argc - 1) << std::endl;
        printHelp(argv[0]);
        return 1;
    }
    if (!boost::filesystem::exists(argv[1])) {
        std::cout << "Error: File " << argv[1] << " does not exist." << std::endl;
        printHelp(argv[0]);
        return 1;
    }
    if (!boost::filesystem::exists(argv[2])) {
        std::cout << "Error: File " << argv[2] << " does not exist." << std::endl;
        printHelp(argv[0]);
        return 1;
    }

    double radar_0_to_1_distance_in_m = 48 * 0.0254;
    TwoRadarsMap2d::Params params;
    params.radar_pos_in_m_[0] = cv::Point2f(0, 0);
    params.radar_pos_in_m_[1] = cv::Point2f((float)radar_0_to_1_distance_in_m, 0);
    params.workarea_map_.map_min_x_in_m_ = -6;
    params.workarea_map_.map_max_x_in_m_ = 6;
    params.workarea_map_.map_min_y_in_m_ = 0;
    params.workarea_map_.map_max_y_in_m_ = 10;
    printParams(argv[1], argv[2], params);

    TwoAmplCsvFilesStream two_radars_stream(argv[1], argv[2]);
    TwoRadarsMap2d radar_map(params);
    cv::namedWindow("Radar map", cv::WINDOW_AUTOSIZE);
    int step_index = 0;
    while (two_radars_stream.moveNext()) {
        radar_map.plotMeas(two_radars_stream.getCurrent());
        cv::imshow("Radar map", radar_map.getColorMap());
        int key = cv::waitKey();
        switch (key) {
        case 27: return 0;
        case 's': {
            std::stringstream strstr;
            strstr << "Radar_map_" << step_index << ".png";
            cv::imwrite(strstr.str(), radar_map.getColorMap());
        }
            break;
        default:
            break;
        }
        ++step_index;
    }
    return 0;
}
