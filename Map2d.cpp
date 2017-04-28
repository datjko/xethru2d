#include "utils.h"

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace pt = boost::property_tree; 
typedef pt::ptree PropTree;

namespace cv {
    inline std::istream& operator >> (std::istream& stream, cv::Point2f& p) {
        return stream >> p.x >> std::skipws >> p.y;
    }

    inline std::ostream& operator << (std::ostream& stream, cv::Point2f const& p) {
        return stream << p.x << " " << p.y;
    }
}

class Map2d {
public:
    double map_min_x_in_m_;
    double map_max_x_in_m_;
    double map_min_y_in_m_;
    double map_max_y_in_m_;
    double pixels_in_one_m_;

    Map2d() {
        map_min_x_in_m_ = -4;
        map_min_y_in_m_ =  0;
        map_max_x_in_m_ =  4;
        map_max_y_in_m_ = 10;
        pixels_in_one_m_ = 50;
    }

    int getWidhthInPixels() const { return (int)(pixels_in_one_m_ * (map_max_x_in_m_ - map_min_x_in_m_)); }
    int getHeightInPixels() const { return (int)(pixels_in_one_m_ * (map_max_y_in_m_ - map_min_y_in_m_)); }

    cv::Size getSizeInPixels() const {
        return cv::Size(getWidhthInPixels(), getHeightInPixels());
    }

    template<typename t_cvPointLike>
    cv::Point2i toPixels(t_cvPointLike const& map_2d_coords) const {
        return cv::Point2i((int)((map_2d_coords.x - map_min_x_in_m_) * pixels_in_one_m_ + 0.5)
            , (int)(getHeightInPixels() - (map_2d_coords.y - map_min_y_in_m_) * pixels_in_one_m_ + 0.5));
    }

    template<typename t_cvPointLike>
    cv::Point2f fromPixels(t_cvPointLike const& pixel_2d_coords) const {
        return cv::Point2f(pixel_2d_coords / pixels_in_one_m_ + map_min_x_in_m_
            , getHeightInPixels() - (pixel_2d_coords.y / pixels_in_one_m_ + map_min_y_in_m_));
    }


    void fromPropTree(PropTree const& prop_tree) {
        map_min_x_in_m_ = prop_tree.get<double>("map_min_x_in_m.<xmlattr>.v");
        map_max_x_in_m_ = prop_tree.get<double>("map_max_x_in_m.<xmlattr>.v");
        map_min_y_in_m_ = prop_tree.get<double>("map_min_y_in_m.<xmlattr>.v");
        map_max_y_in_m_ = prop_tree.get<double>("map_max_y_in_m.<xmlattr>.v");
        pixels_in_one_m_ = prop_tree.get<double>("pixels_in_one_m.<xmlattr>.v");
    }
    void toPropTree(PropTree& prop_tree) const {
        prop_tree.put("map_min_x_in_m.<xmlattr>.v", map_min_x_in_m_);
        prop_tree.put("map_max_x_in_m.<xmlattr>.v", map_max_x_in_m_);
        prop_tree.put("map_min_y_in_m.<xmlattr>.v", map_min_y_in_m_);
        prop_tree.put("map_max_y_in_m.<xmlattr>.v", map_max_y_in_m_);
        prop_tree.put("pixels_in_one_m.<xmlattr>.v", pixels_in_one_m_);
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
            p[col] = pixel_distance_functor(distance_in_pixels);
        }
    }
}

struct RadarRawMeas {
    struct Params {
        double first_bin_offset_in_m_;
        double bin_size_in_m_;

        Params()
            : first_bin_offset_in_m_(0.0514) //fixme?
            , bin_size_in_m_(0.0514) //fixme?
        {}
    };

    struct Bin {
        double amplitude_;

        Bin(double amplitude = 0) : amplitude_(amplitude) {}
    };

    typedef std::vector<Bin> Bins;

    RadarRawMeas() : radar_index_(0) {}

    std::size_t radar_index_; //fixme: used for debug only
    Bins bins_;
};

struct RadarAdaptiveClutterMap : RadarRawMeas {
    struct Params {
        double adaptive_factor_;
        std::size_t spread_max_step_;

        Params()
            : adaptive_factor_(0.05)
            , spread_max_step_(2)
        {}
    };

    Params params_;

    RadarAdaptiveClutterMap(Params const& p = Params()) : params_(p) {}

    void setFirst(Bins const& raw_bins) { bins_ = raw_bins; }

    void apply(Bins& in_out_raw_bins) {
        if (bins_.size() != in_out_raw_bins.size()) {
            //fixme? is this case ever happen?
            //: reset the map
            setFirst(in_out_raw_bins);
            in_out_raw_bins = Bins(in_out_raw_bins.size());
            return;
        }
        //: Update the adaptive cluttermap bins and calculate the result in_out_raw_bins:
        for (std::size_t i = 0; i < in_out_raw_bins.size(); ++i) {
            bins_[i].amplitude_ = bins_[i].amplitude_ * (1 - params_.adaptive_factor_) + in_out_raw_bins[i].amplitude_ * params_.adaptive_factor_;
            in_out_raw_bins[i].amplitude_ -= bins_[i].amplitude_;
        }
        if (params_.spread_max_step_ > 0) {
            Bins in_out_raw_bins_copy = in_out_raw_bins;
            for (std::size_t i = 0; i < in_out_raw_bins.size(); ++i) {
                std::size_t j_min = (i > params_.spread_max_step_) ? i - params_.spread_max_step_ : 0;
                std::size_t j_max = std::min(i + params_.spread_max_step_ + 1, in_out_raw_bins.size());
                double a = in_out_raw_bins_copy[i].amplitude_;
                for (std::size_t j = j_min; j < j_max; ++j)
                    a = std::max(a, in_out_raw_bins_copy[j].amplitude_);
                in_out_raw_bins[i].amplitude_ = a;
            }
        }
    }
};

typedef std::vector<RadarRawMeas> RadarRawMeasSet;

class RadarRawMeasSetStreamBase {
public:
    virtual ~RadarRawMeasSetStreamBase() {}
    virtual bool moveNext() = 0;
    virtual RadarRawMeasSet const& getCurrent() const = 0;
};

#define XETHRU_DEFAULT_PRESENCE_THRESHOLD 0.0002

class SingleRadarMap2d {
public:

    struct SharedParams {
        Map2d workarea_map_;
        RadarRawMeas::Params radar_raw_frame_params_;
        double min_amplitude_value_;
        double max_amplitude_value_;

        SharedParams()
            : min_amplitude_value_(XETHRU_DEFAULT_PRESENCE_THRESHOLD / 10)
            , max_amplitude_value_(XETHRU_DEFAULT_PRESENCE_THRESHOLD * 2)
        {}

        void fromPropTree(PropTree const& prop_tree) {
            workarea_map_.fromPropTree(prop_tree.get_child("workarea_map"));
            min_amplitude_value_ = prop_tree.get<double>("min_amplitude.<xmlattr>.v");
            max_amplitude_value_ = prop_tree.get<double>("max_amplitude.<xmlattr>.v");
        }
        void toPropTree(PropTree& prop_tree) const {
            workarea_map_.toPropTree(prop_tree.add("workarea_map", ""));
            prop_tree.put("min_amplitude.<xmlattr>.v", min_amplitude_value_);
            prop_tree.put("max_amplitude.<xmlattr>.v", max_amplitude_value_);
        }
    };

    struct Params {
        SharedParams shared_params_;
        cv::Point2f radar_pos_in_m_;

        Params() {}

        Params(SharedParams const& shared_params, cv::Point2f radar_pos_in_m)
            : shared_params_(shared_params)
            , radar_pos_in_m_(radar_pos_in_m)
        {}

        void fromPropTree(PropTree const& prop_tree) {
            shared_params_.fromPropTree(prop_tree.get_child("shared_params"));
            radar_pos_in_m_ = prop_tree.get<cv::Point2f>("radar.<xmlattr>.pos_in_m");
        }
        void toPropTree(PropTree& prop_tree) const {
            shared_params_.toPropTree(prop_tree.add("shared_params", ""));
            prop_tree.put("radar.<xmlattr>.pos_in_m", radar_pos_in_m_);
        }
    };

    SingleRadarMap2d() {}

    SingleRadarMap2d(Params const& params) : params_(params) {
        onAfterParamsUpdate();
    }

    Params const& getParams() const { return params_; }

    void setParams(Params const& params) {
        params_ = params;
        onAfterParamsUpdate();
    }

    cv::Mat const& getGreyMap() const { return grey_map_; }

    cv::Mat const& plotMeas(RadarRawMeas const& radar_raw_meas) {
        std::vector<uchar> amplitudes_as_uchars_lut(256, 0);
        double amplitude_scale_to_uchar = 255. / (params_.shared_params_.max_amplitude_value_ - params_.shared_params_.min_amplitude_value_);
        for (std::size_t i = 0; i < std::min(std::size_t(256), radar_raw_meas.bins_.size()); ++i) {
            double ampl_value = std::min(std::max(radar_raw_meas.bins_[i].amplitude_, params_.shared_params_.min_amplitude_value_), params_.shared_params_.max_amplitude_value_);
            amplitudes_as_uchars_lut[i] = (uchar)((ampl_value - params_.shared_params_.min_amplitude_value_) * amplitude_scale_to_uchar);
        }
        cv::LUT(precalculated_radar_distance_map_in_bins_, amplitudes_as_uchars_lut, grey_map_);
        //cv::imshow(std::string("radar-") + (char)('0' + radar_raw_meas.radar_index_), grey_map_);
        return grey_map_;
    }

private:
    void onAfterParamsUpdate() {
        pix2bins_converter_ = Pixels2BinsDistanceConverter(params_.shared_params_);

        precalculated_radar_distance_map_in_bins_.create(params_.shared_params_.workarea_map_.getSizeInPixels());
        cv::Point radar_pos_in_pix = params_.shared_params_.workarea_map_.toPixels(params_.radar_pos_in_m_);
        makePointDistanceMatrix(precalculated_radar_distance_map_in_bins_, radar_pos_in_pix, pix2bins_converter_);

        grey_map_.create(precalculated_radar_distance_map_in_bins_.size());
    }

private:
    struct Pixels2BinsDistanceConverter {
        double offset_in_pixels_;
        double bins_in_one_pixel_;

        Pixels2BinsDistanceConverter() : offset_in_pixels_(0), bins_in_one_pixel_(0) {}

        Pixels2BinsDistanceConverter(SharedParams const& shared_params) {
            offset_in_pixels_ = shared_params.radar_raw_frame_params_.first_bin_offset_in_m_ * shared_params.workarea_map_.pixels_in_one_m_;
            bins_in_one_pixel_ = 1. / (shared_params.workarea_map_.pixels_in_one_m_ * shared_params.radar_raw_frame_params_.bin_size_in_m_);
        }

        uchar operator()(double distance_in_pixels) const {
            return (distance_in_pixels >= offset_in_pixels_)
                ? (uchar)((distance_in_pixels - offset_in_pixels_) * bins_in_one_pixel_ + 0.5)
                : uchar(0);
        }
    };

    Params params_;

    Pixels2BinsDistanceConverter pix2bins_converter_;

    cv::Mat_<uchar> precalculated_radar_distance_map_in_bins_;
    cv::Mat_<uchar> grey_map_;
};


class MultiRadarMap2d {
public:

    struct Params {
        SingleRadarMap2d::SharedParams shared_params_;
        std::vector<cv::Point2f> radar_pos_in_m_;

        Params() {}

        void fromPropTree(PropTree const& prop_tree) {
            shared_params_.fromPropTree(prop_tree.get_child("shared_params"));
            BOOST_FOREACH(pt::ptree::value_type const& v, prop_tree.get_child("radars")) {
                radar_pos_in_m_.push_back(v.second.get<cv::Point2f>("<xmlattr>.pos_in_m"));
            }
        }
        void toPropTree(PropTree& prop_tree) const {
            shared_params_.toPropTree(prop_tree.add("shared_params", ""));
            PropTree& radars_prop_tree = prop_tree.add("radars", "");
            BOOST_FOREACH(cv::Point2f const& v, radar_pos_in_m_) {
                radars_prop_tree.add("radar", "").put("<xmlattr>.pos_in_m", v);
            }
        }
    };

    MultiRadarMap2d(Params const& params) : params_(params) {
        onAfterParamsUpdate();
    }

    cv::Mat getColorMap() const { return color_map_; }

    cv::Mat_<uchar> getHeatMap() const { return heat_map_; }

    Params const& getParams() const { return params_; }

    void setParams(Params const& params) {
        params_ = params;
        onAfterParamsUpdate();
    }

    void plotMeas(RadarRawMeasSet const& meas) {
        for (std::size_t radar_index = 0; radar_index < meas.size(); ++radar_index) {
            cv::Mat_<uchar> const& single_radar_heat_map = single_radar_maps_[radar_index].plotMeas(meas[radar_index]);
            std::size_t channel_index = radar_index % 3;

            if (radar_index < 3)
                color_map_channels_[channel_index] = single_radar_heat_map;
            else
                color_map_channels_[channel_index] = cv::max(color_map_channels_[channel_index], single_radar_heat_map);

            if (radar_index == 0)
                heat_map_ = single_radar_heat_map.clone();
            else
                heat_map_ = cv::min(heat_map_, single_radar_heat_map);
        }

        cv::merge(color_map_channels_, 3, color_map_);
    }

private:
    void onAfterParamsUpdate() {
        makeMats(params_.shared_params_.workarea_map_.getSizeInPixels());

        single_radar_maps_.resize(params_.radar_pos_in_m_.size());
        for (std::size_t i = 0; i < params_.radar_pos_in_m_.size(); ++i)
            single_radar_maps_[i].setParams(SingleRadarMap2d::Params(params_.shared_params_, params_.radar_pos_in_m_[i]));
    }

    void makeMats(cv::Size map_size_in_pixels) {
        for (std::size_t i = 0; i < 3; ++i)
            //color_map_channels_[i].create(map_size_in_pixels);
            color_map_channels_[i] = cv::Mat_<uchar>::zeros(map_size_in_pixels);
        color_map_.create(map_size_in_pixels);
        heat_map_.create(map_size_in_pixels);
    }

private:
    Params params_;

    std::vector<SingleRadarMap2d> single_radar_maps_;
    cv::Mat_<uchar> color_map_channels_[3];
    cv::Mat_<cv::Vec3b> color_map_;
    cv::Mat_<uchar> heat_map_;
};



class AmplCsvFilesStream : public RadarRawMeasSetStreamBase {
public:
    struct Params {
        std::vector<std::string> files_;

        Params() {}

        void fromPropTree(PropTree const& prop_tree) {
            BOOST_FOREACH(pt::ptree::value_type const& v, prop_tree) {
                files_.push_back(v.second.get<std::string>("<xmlattr>.path"));
            }
        }
        void toPropTree(PropTree& prop_tree) const {
            BOOST_FOREACH(std::string const& v, files_) {
                prop_tree.add("file", "").put("<xmlattr>.path", v);
            }
        }
    };

    AmplCsvFilesStream(Params const& params)
        : radar_ampl_files_(params.files_.size())
        , current_meas_set_(radar_ampl_files_.size())
    {
        for (std::size_t i = 0; i < params.files_.size(); ++i) {
            radar_ampl_files_[i].open(params.files_[i].c_str());
            current_meas_set_[i].radar_index_ = i;
        }
    }

    virtual bool moveNext() {
        bool ok = true;
        std::string csv_line;
        for (std::size_t i = 0; ok && i < radar_ampl_files_.size(); ++i) {
            ok = ok && radar_ampl_files_[i].good();
            ok = ok && std::getline(radar_ampl_files_[i], csv_line);
            if (ok)
                readCsvLine(current_meas_set_[i], csv_line, 1);
        }
        return ok;
    }

    virtual RadarRawMeasSet const& getCurrent() const { return current_meas_set_; }

protected:
    RadarRawMeasSet& getMutableCurrent() { return current_meas_set_; }

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
    std::vector<std::ifstream> radar_ampl_files_;
    RadarRawMeasSet current_meas_set_;
};

class FilteredAmplCsvFilesStream : public AmplCsvFilesStream {
public:
    struct Params : AmplCsvFilesStream::Params {
        RadarAdaptiveClutterMap::Params clutter_map_params_;

        Params() {}

        void fromPropTree(PropTree const& prop_tree) {
            clutter_map_params_.adaptive_factor_ = prop_tree.get("<xmlattr>.adaptive_factor", clutter_map_params_.adaptive_factor_);
            AmplCsvFilesStream::Params::fromPropTree(prop_tree);
        }
        void toPropTree(PropTree& prop_tree) const {
            prop_tree.put("<xmlattr>.adaptive_factor", clutter_map_params_.adaptive_factor_);
            AmplCsvFilesStream::Params::toPropTree(prop_tree);
        }
    };

    FilteredAmplCsvFilesStream(Params const& params)
        : AmplCsvFilesStream(params)
        , radar_clutter_maps_(params.files_.size())
    {
        //: read first record for each radar and initialize the radar_clutter_maps_:
        bool ok = AmplCsvFilesStream::moveNext();
        if (ok) {
            for (std::size_t i = 0; i < radar_clutter_maps_.size(); ++i) {
                radar_clutter_maps_[i].setFirst(getCurrent()[i].bins_);
            }
        }
    }

    virtual bool moveNext() {
        bool ok = AmplCsvFilesStream::moveNext();
        if (ok) {
            for (std::size_t i = 0; i < radar_clutter_maps_.size(); ++i) {
                radar_clutter_maps_[i].apply(getMutableCurrent()[i].bins_);
            }
        }
        return ok;
    }

private:
    std::vector<RadarAdaptiveClutterMap> radar_clutter_maps_;
};


struct Params : MultiRadarMap2d::Params {
    FilteredAmplCsvFilesStream::Params radar_params_;
    int wait_between_frames_in_ms_;

    Params() : wait_between_frames_in_ms_(330) {}
    Params(PropTree const& prop_tree) : wait_between_frames_in_ms_(330) {
        fromPropTree(prop_tree);
    }

    void fromPropTree(PropTree const& prop_tree) {
        MultiRadarMap2d::Params::fromPropTree(prop_tree);
        radar_params_.fromPropTree(prop_tree.get_child("files"));
        wait_between_frames_in_ms_ = prop_tree.get("wait_between_frames_in_ms.<xmlattr>.v", wait_between_frames_in_ms_);
    }
    void toPropTree(PropTree& prop_tree) const {
        MultiRadarMap2d::Params::toPropTree(prop_tree);
        radar_params_.toPropTree(prop_tree.add("files", ""));
        prop_tree.put("wait_between_frames_in_ms.<xmlattr>.v", wait_between_frames_in_ms_);
    }

    void addRadar(cv::Point2f pos_in_m, std::string const& path_to_ampl_file) {
        MultiRadarMap2d::Params::radar_pos_in_m_.push_back(pos_in_m);
        radar_params_.files_.push_back(path_to_ampl_file);
    }
};

void printParams(Params const& params) {
    PropTree prop_tree;
    params.toPropTree(prop_tree.add("xethru_map2d", ""));
    pt::write_xml(std::cout, prop_tree, pt::xml_writer_make_settings(' ', 2));
}

void printExampleParams() {
    std::cout << "Example parameters:" << std::endl;
    std::cout << "------ COPY FROM NEXT LINE -----" << std::endl;
    PropTree prop_tree;
    Params params;
    params.addRadar(cv::Point2f(0, 0), "path/to/origin_amplitude_0.csv");
    float radar_0_to_1_distance_in_m = (float)(48 * 0.0254);
    params.addRadar(cv::Point2f(radar_0_to_1_distance_in_m, 0), "path/to/endpoint_amplitude_0.csv");
    printParams(params);
}



void play(Params const& params) {
    std::cout << "Starting with following parameters:" << std::endl;
    printParams(params);
    FilteredAmplCsvFilesStream multiradars_stream(params.radar_params_);
    MultiRadarMap2d radars_map(params);
    cv::namedWindow("Radars map", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Radars heat map", cv::WINDOW_AUTOSIZE);
    int step_index = 0;
    while (multiradars_stream.moveNext()) {
        radars_map.plotMeas(multiradars_stream.getCurrent());
        cv::imshow("Radars map", radars_map.getColorMap());
        cv::imshow("Radars heat map", radars_map.getHeatMap());
        int key = cv::waitKey(params.wait_between_frames_in_ms_);
        switch (key) {
        case 27: return;
        case 's': {
                std::stringstream strstr;
                strstr << "Radar_map_" << step_index << ".png";
                cv::imwrite(strstr.str(), radars_map.getColorMap());
                strstr.swap(std::stringstream());
                strstr << "Radar_heat_map_" << step_index << ".png";
                cv::imwrite(strstr.str(), radars_map.getHeatMap());
            }
            break;
        default:
            break;
        }
        ++step_index;
    }
}

void play(PropTree const& prop_tree) {
    play(Params(prop_tree.get_child("xethru_map2d")));
}

void printHelp(char const* program_name) {
    std::cout << "Usage:\n"
        << program_name << " path_to_parameters.xml\n"
        << "or\n"
        << program_name << " path_to_parameters.info\n"
        << std::endl;
    printExampleParams();
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Error: Incorrect number of arguments: " << (argc - 1) << std::endl;
        printHelp(argv[0]);
        return 1;
    }
    boost::filesystem::path path_to_params_file(argv[1]);
    if (!boost::filesystem::exists(path_to_params_file)) {
        std::cout << "Error: File " << argv[1] << " does not exist." << std::endl;
        printHelp(argv[0]);
        return 1;
    }

    PropTree prop_tree;
    std::string file_ext = path_to_params_file.extension().string();
    if (file_ext == ".xml")
        pt::read_xml(path_to_params_file.string(), prop_tree);
    else if (file_ext == ".info")
        pt::read_info(path_to_params_file.string(), prop_tree);
    else {
        std::cout << "Error: Unknown file extension '" << file_ext << "'. Should be either .xml or .info." << std::endl;
        printHelp(argv[0]);
        return 1;
    }

    play(prop_tree);
    return 0;
}
