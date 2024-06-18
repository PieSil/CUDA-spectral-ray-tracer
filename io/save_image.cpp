#include "save_image.h"
#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

void save_img(uchar_img& image, const char* filename) {
   
	fs::path p("renders/" + std::string(filename));
	fs::create_directories(p.parent_path());
	image.save(p.string().c_str());
	std::clog << "Image saved as: " << p.filename() << std::endl;
}