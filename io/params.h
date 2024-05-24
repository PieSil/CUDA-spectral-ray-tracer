#ifndef SPECTRAL_RT_PROJECT_PARAMS_H
#define SPECTRAL_RT_PROJECT_PARAMS_H

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

using namespace std;

typedef unsigned int uint;

#define CORNELL 0
#define PRISM 1
#define TRIS 2

static const string sceneIdToStr[] = {"Cornell Box", "Prism World", "Triangles"};

class parameters {
public:

	parameters() {
		resetYres();
	}

	const string getImgTitle() const {
		auto retval = image_title.empty() ? string(sceneIdToStr[scene]) : image_title;
		return retval;
	}

	const string getLogSubdir() const {
		return log_subdir;
	}

	const uint getSceneId() const {
		return scene;
	}

	const uint getXres() const {
		return xres;
	}

	const uint getYres() const {
		return yres;
	}

	const float getAR() const {
		return ar;
	}

	const uint getXcsize() const {
		uint retval = xcsize == 0 ? ycsize : xcsize;
		retval = retval == 0 ? xres : retval;
		return retval;
	}

	const uint getYcsize() const {
		uint retval = ycsize == 0 ? xcsize : ycsize;
		retval = retval == 0 ? yres : retval;
		return retval;
	}

	const uint getNSamples() const {
		return n_samples;
	}

	const uint getBounceLimit() const {
		return bounce_limit;
	}

	const bool logActive() const {
		return do_log;
	}

	const bool doSaveImage() const {
		return do_save;
	}

	const bool showRender() const {
		return show_render;
	}

	void const setImgTitle(const string val_str) {
		image_title = val_str;
	}

	void const setLogSubdir(const string val_str) {
		log_subdir = val_str;
	}

	void const setScene(const string val_str) {
		try {
			scene = stoul(val_str);
		}
		catch (...) {
			cerr << "Error while parsing scene arg value, keeping previous (default most likely) value" << endl;
		}
	}

	void const setXres(const string val_str) {
		try {
			xres = stoul(val_str);
			resetYres();
		}
		catch (...) {
			cerr << "Error while parsing xres arg value, keeping previous (default most likely) value" << endl;
		}
	}

	void const setAR(const string val_str) {

		try {
			ar = parseAR(val_str);
			resetYres();
		}
		catch (...) {
			cerr << "Error while parsing aspect-ratio arg value, keeping previous (default most likely) value" << endl;
		}
	}

	void const setXcsize(const string val_str) {

		try {
			xcsize = stoul(val_str);
		}
		catch (...) {
			cerr << "Error while parsing xcsize arg value, keeping previous (default most likely) value" << endl;
		}
	}

	void const setYcsize(const string val_str) {

		try {
			ycsize = stoul(val_str);
		}
		catch (...) {
			cerr << "Error while parsing ycsize arg value, keeping previous (default most likely) value" << endl;
		}
	}

	void const setNSamples(const string val_str) {

		try {
			n_samples = stoul(val_str);
		}
		catch (...) {
			cerr << "Error while parsing n_samples arg value, keeping previous (default most likely) value" << endl;
		}
	}

	void const setBounceLimit(const string val_str) {

		try {
			bounce_limit = stoul(val_str);
		}
		catch (...) {
			cerr << "Error while parsing bounce-limit arg value, keeping previous (default most likely) value" << endl;
		}
	}

	void const setDoLog(const bool val) {
		do_log = val;
	}

	void const setShowRender(const bool val) {
		show_render = val;
	}

	void const setDoSave(const bool val) {
		do_save = val;
	}

private:
	void resetYres() {
		/* Calculate the image height, and ensure that it's at least 1 */
		yres = static_cast<uint>(xres / ar);
		yres = (yres < 1) ? 1 : yres;
	}

	float parseAR(string ar_string) {
		stringstream ssar(ar_string);
		string num_string;
		getline(ssar, num_string, '/');
		float ar = stof(num_string);
		if (getline(ssar, num_string, '/')) {
			ar /= stof(num_string);
			if (getline(ssar, num_string, '/')) {
				cout << "Characters inserted after aspect ratio's denominator will be ignored, computed AR value is: " << ar << endl;
			}
		}

		return ar;
	}

	//output
	string image_title;

	//logs
	string log_subdir = "";

	//world
	uint scene = CORNELL;

	//image
	uint xres = 600;
	uint yres;
	float ar = 1.0f;

	//chunks
	uint xcsize = 0;
	uint ycsize = 0;

	//quality
	uint n_samples = 500;
	uint bounce_limit = 10;

	//logs
	bool do_log = false;
	bool show_render = true;
	bool do_save = false;
};

//a class to ease parsing and managing of CLI arguments
class param_manager {
public:
	static shared_ptr<param_manager> getInstance() {
		if (instance.get() == nullptr) {
			instance = make_shared<param_manager>(param_manager());
		}

		return instance;
	}

	void parseArgs(int argc, char* argv[]) {


		/*
		 * -t/--title: name of the image file (without extension, forced to .bmp for now
		 * -lsub/--log-subdir: subdirectory where to store log file (relative to ./logs/)
		 * -s/--scene: id of selected scene as uint value, default: 0 (for now)
		 * -xr/--xres: image resolution along x, default: 600
		 * -yr/--yres: NOT MANAGED, determined by xres and ar
		 * -ar/--aspect-ratio: aspect ratio of the scene, expressed as a single value or two values separated by /, default: 1.0
		 * -xc/--xcsize: chunk resolution along x, default = yc (if not set then xres)
		 * -yc/--ycsize: chunk resolution along y, default = xc (if not set then yres)
		 * -ns/--nsamples: number of samples per pixel, default: 500
		 * -bl/--bounce-limit: max number of bounces for each ray, default: 10
		 * --do-log: enables logging at the end of render
		 * --no-show: disables display of render
		 * --save: saves final result as image
		 * 
		 * -- POSSIBLE FUTURE SUPPORTED ARGUMENTS--
		 * -nstr/--nstreams: number of different streams to use for rendering, default: 1
		 * -sm/--sort-materials: whether to sort materials based on id after each bounce simulation, default: true
		 * -um/--unified-mem: whether to use unified memory to manage frame buffer accessed by the device
		 * other parameters to manage logging
		 */

		for (int i = 1; i < argc; i++) {
			std::string arg(argv[i]);
			bool is_last = i + 1 == argc;
			if (!is_last && (arg == "-t" || arg == "--title")) {
				params.setImgTitle(argv[++i]);
			}
			else if (!is_last && (arg == "-lsub" || arg == "--log-subdir")) {
				params.setLogSubdir(argv[++i]);
			}
			else if (!is_last && (arg == "-s" || arg == "--scene")) {
				params.setScene(argv[++i]);
			}
			else if (!is_last && (arg == "-xr" || arg == "--xres")) {
				params.setXres(argv[++i]);
			}
			else if (!is_last && (arg == "-ar" || arg == "--aspect-ratio")) {
				params.setAR(argv[++i]);
			}
			else if (!is_last && (arg == "-xc" || arg == "--xcsize")) {
				params.setXcsize(argv[++i]);
			}
			else if (!is_last && (arg == "-yc" || arg == "--ycsize")) {
				params.setYcsize(argv[++i]);
			}
			else if (!is_last && (arg == "-ns" || arg == "--nsamples")) {
				params.setNSamples(argv[++i]);
			}
			else if (!is_last && (arg == "-bl" || arg == "--bounce-limit")) {
				params.setBounceLimit(argv[++i]);
			}
			else if (arg == "--do-log") {
				params.setDoLog(true);
			}
			else if (arg == "--no-show") {
				params.setShowRender(false);
			}
			else if (arg == "--save") {
				params.setDoSave(true);
			}
			else {
				cout << "Unkown argument name: " << arg << endl;
			}
		}
	}

	const parameters getParams() const {
		return params;
	}

private:
	param_manager() {}

	static shared_ptr<param_manager> instance;
	parameters params;
};

#endif