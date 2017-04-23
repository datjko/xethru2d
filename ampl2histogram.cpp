#include "utils.h"




struct AmplHistogram {
    typedef unsigned int BinIndex;
    typedef std::map<BinIndex, std::size_t> Histogram;

    double bins_per_unit_;
    Histogram h_;

    AmplHistogram(double bins_per_unit = 1e9) : bins_per_unit_(bins_per_unit) {}

    void add(double a) {
        BinIndex bi = (BinIndex)(a * bins_per_unit_ + .5);
        Histogram::iterator i = h_.find(bi);
        if (i == h_.end())
            h_.insert(Histogram::value_type(bi, 1));
        else
            ++i->second;
    }
};

struct AmplHistograms {
    typedef std::vector<AmplHistogram> Histograms;
    typedef std::vector<double> StrangeValues;

    double min_, max_, sum_;
    std::size_t num_;
    Histograms h_;
    StrangeValues sv_;

    template<typename t_BpuValuesItr>
    AmplHistograms(t_BpuValuesItr start, t_BpuValuesItr end)
        : min_(1), max_(0), sum_(0)
        , num_(0)
        , h_(start, end) {}

    void add(double a) {
        if (a < 0 || a >= 1) {
            sv_.push_back(a);
            return;
        }
        ++num_;
        sum_ += a;
        if (a < min_)
            min_ = a;
        if (a > max_)
            max_ = a;
        BOOST_FOREACH(AmplHistogram& h, h_) {
            h.add(a);
        }
    }
};


template<typename t_Result>
void readCsvLine(t_Result& res, std::string const& csv_line, int start_field_index) {
    std::size_t field_start_pos = 0;
    std::size_t field_end_pos = 0;
    std::size_t field_index = 0;

    //: skip first start_field_index fields:
    while (field_index < start_field_index && (field_end_pos = csv_line.find(',', field_end_pos + 1)) != std::string::npos)
        ++field_index;
    if (field_end_pos == std::string::npos) {
        //: empty list of bins:
        return;
    }

    field_index = 0;
    do {
        field_start_pos = field_end_pos + 1;
        field_end_pos = csv_line.find(',', field_start_pos);
        std::stringstream strstr;
        if (field_end_pos != std::string::npos)
            strstr << csv_line.substr(field_start_pos, field_end_pos - field_start_pos);
        else
            strstr << csv_line.substr(field_start_pos);
        double a;
        strstr >> a;
        res.add(a);
        ++field_index;
    } while (field_end_pos != std::string::npos);
}

template<typename t_Result>
void readCsvStream(t_Result& res, std::istream& csv_istream, int start_field_index) {
    std::string csv_line;
    while (std::getline(csv_istream, csv_line))
        readCsvLine(res, csv_line, start_field_index);
}

template<typename t_Result>
void readCsvFile(t_Result& res, fs::path const& csv_file_path, int start_field_index) {
    std::ifstream is(csv_file_path.c_str());
    readCsvStream(res, is, start_field_index);
}


void print(AmplHistogram const& h, std::ostream& ostream) {
    BOOST_FOREACH(AmplHistogram::Histogram::value_type const& v, h.h_) {
        ostream << std::scientific << v.first / h.bins_per_unit_ << " " << v.second << "\n";
    }
}

void print(AmplHistograms const& h, std::ostream& ostream) {
    ostream << "num: " << h.num_ << "\n"
        << "min: " << h.min_ << "\n"
        << "max: " << h.max_ << "\n"
        << "avg: " << h.sum_ / h.num_ << "\n"
        << "\n";
    if (!h.sv_.empty()) {
        ostream << "sv:\n";
        BOOST_FOREACH(AmplHistograms::StrangeValues::value_type const& v, h.sv_) {
            ostream << v << "\n";
        }
        ostream << "\n";
    }

    ostream << "Histograms sunnary:\n"
        << "bpu, size, @25%, @50%, @75%\n";
    BOOST_FOREACH(AmplHistograms::Histograms::value_type const& v, h.h_) {
        AmplHistogram::Histogram::const_iterator i = v.h_.begin();
        std::size_t s = v.h_.size();
        ostream << v.bins_per_unit_ << ", " << s << ", ";
        std::advance(i, s / 4);
        ostream << i->first / v.bins_per_unit_ << ", ";
        std::advance(i, s / 4);
        ostream << i->first / v.bins_per_unit_ << ", ";
        std::advance(i, s / 4);
        ostream << i->first / v.bins_per_unit_ << "\n";
    }
    ostream << "\n";

    ostream << "Histograms:\n";
    BOOST_FOREACH(AmplHistograms::Histograms::value_type const& v, h.h_) {
        print(v, ostream);
        ostream << "\n";
    }
}

template<typename t_Result>
void printToFile(t_Result const& res, fs::path const& file_path) {
    std::ofstream os(file_path.c_str());
    print(res, os);
}


cv::Mat_<cv::Vec3b> plot(AmplHistogram const& h, std::size_t total_num_of_items, double max_value = 0.062, double max_bin_height_to_total = 0.0025) {
    int num_of_bins = (int)(max_value * h.bins_per_unit_ + .5);
    cv::Size size(num_of_bins, 100);
    cv::Mat_<cv::Vec3b> res(size, cv::Vec3b(0, 0, 0));

    cv::Point bar_bottom_pos(0, size.height - 1);
    BOOST_FOREACH(AmplHistogram::Histogram::value_type const& v, h.h_) {
        bar_bottom_pos.x = v.first; // (int)(double(v.first) / num_of_bins + .5);
        int bar_height = (int)(v.second / (total_num_of_items * max_bin_height_to_total) + .5);
        cv::line(res, bar_bottom_pos, cv::Point(bar_bottom_pos.x, bar_bottom_pos.y - bar_height), cv::Scalar(0, 0, 255));
    }
    return res;
}


void processFile(fs::path const& csv_file_path, fs::path const& output_folder_path, int start_field_index) {
    static const double c_bins_per_unit[] = {
        //1e9, 1e8,
        //1e7, 1e6,
        //1e5,
        1e4,
    };
    AmplHistograms h(c_bins_per_unit, c_bins_per_unit + sizeof(c_bins_per_unit) / sizeof(c_bins_per_unit[0]));
    readCsvFile(h, csv_file_path, start_field_index);
    printToFile(h, output_folder_path / csv_file_path.filename());

    //cv::imshow("Amplitudes histogram", plot(h.h_[0], h.num_));
    //cv::waitKey();
    cv::imwrite((output_folder_path / csv_file_path.filename()).string() + ".png", plot(h.h_[0], h.num_));
}

void processFiles(std::vector<std::string> const& csv_file_paths, fs::path const& output_folder_path, int start_field_index) {
    if (!fs::exists(output_folder_path))
        fs::create_directories(output_folder_path);
    BOOST_FOREACH(std::string const& csv_file_path, csv_file_paths) {
        processFile(csv_file_path, output_folder_path, start_field_index);
    }
}

void printHelp(char const* program_name) {
    std::cout << "Usage:\n"
        << program_name << " path_to_radar_csv_files_folder masks output_folder_path" << std::endl;
}


int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Error: Incorrect number of arguments: " << (argc - 1) << std::endl;
        printHelp(argv[0]);
        return 1;
    }
    if (!boost::filesystem::exists(argv[1])) {
        std::cout << "Error: path " << argv[1] << " does not exist." << std::endl;
        printHelp(argv[0]);
        return 1;
    }

    std::vector<std::string> paths;
    if (fs::is_directory(argv[1]))
        paths = findFilesPaths(argv[1], argv[2]);
    else
        paths.push_back(argv[1]);

    processFiles(paths, argv[3], 1);
    return 0;
}
