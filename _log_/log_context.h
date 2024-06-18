#ifndef SPECTRAL_RT_PROJECT_LOG_CONTEXT_H
#define SPECTRAL_RT_PROJECT_LOG_CONTEXT_H

#ifdef WIN32
#define OS_SEP "\\"
#else
#define OS_SEP "/"
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <unordered_map>
#include <set>
#include <vector>
#include <memory>
#include <filesystem>
#include "utility.h"

using namespace std;
namespace fs = std::filesystem;

enum FilenameOption {
	EMPTY_UNDERSCORE,
	TIMESTAMP,
	TITLE
};

//A class to store log data about the application's performance and eventually save it to file
class log_context {
public:
	static std::shared_ptr<log_context> getInstance() {
		if (instance.get() == nullptr) {
			instance = make_shared<log_context>(log_context());
		}

		return instance;
	}

	void to_file();

	string build_file_content();

	string build_filename();

	void add_filename_option(const FilenameOption opt);

	void setPath(const string path_str) {
		if (!path_str.empty()) {
			rel_path = fs::path(path_str);
		}
	}

	void setFilename(const string new_filename) {
		if (!new_filename.empty()) {
			string new_path = rel_path.parent_path().string();

			new_path.append(OS_SEP).append(new_filename);
			setPath(new_path);
		}
	}

	void setDir(const string new_dir) {
		if (!new_dir.empty()) {
			string new_path = new_dir;
			string file = rel_path.filename().string();

			new_path.append(OS_SEP).append(file);
			setPath(new_path);
		}
	}

	void append_dir(const string dirname) {
		if (!dirname.empty()) {
			string new_dir = rel_path.parent_path().string();
			new_dir.append(OS_SEP).append(dirname);

			setDir(new_dir);
		}
	}

	void add_title(string _title);

	void add_entry(string name, string value);

	void add_entry(string name, unsigned int value);

	void add_entry(string name, size_t value);

	void add_entry(string name, int value);

	//std::numeric_limits<float>::digits10 + 1

	void add_entry(string name, float value);

	void add_entry(string name, double value);

    void sum_value(string name, float value);


private:
	log_context() {
		setPath("logs/log.txt");
	}

	string title; //a human-readable id which will always be prepended to the string set as filename of the path
	static std::shared_ptr<log_context> instance;
	fs::path rel_path;
	vector<string> data_insertion_order;
	set<FilenameOption> name_options;
	unordered_map<string, string> data;
};

#endif