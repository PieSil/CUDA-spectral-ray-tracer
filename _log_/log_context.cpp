#include "log_context.h"

std::shared_ptr<log_context> log_context::instance;

void log_context::to_file() {
	fs::path dir_path = rel_path.parent_path();
	fs::create_directories(dir_path);

	string file_path = dir_path.string().append(OS_SEP).append(build_filename());
	std::ofstream outputFile(file_path);

	// Check if file opened successfully
	if (outputFile.is_open()) {
		// Write content to the file
		outputFile << build_file_content();
		// Close the file
		outputFile.close();
		std::clog << "Log file saved successfully at: " << file_path << std::endl;
	}
	else {
		std::cerr << "Failed to save log file at: " << file_path << std::endl;
	}
}

string log_context::build_file_content() {
	string content;
	for (auto const& name : data_insertion_order) {
		string entry = name;
		string val = data[name];
		entry.append(": ").append(val).append("\n");
		content.append(entry);
	}

	return content;
}

string log_context::build_filename() {
	string file_name;
	for (auto i = name_options.begin(); i != name_options.end(); i++) {
		switch (*i) {
		case FilenameOption::TIMESTAMP:
		{
			//get a timestamp
			auto t = std::chrono::system_clock::now();
			file_name.append(to_string(std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count()));
		}

		break;

		case FilenameOption::TITLE:
			file_name.append(title);
			break;

		default:
			break;
		}

		file_name.append("_");
	}

	file_name.append(rel_path.filename().string());
	return file_name;
}

void log_context::add_filename_option(FilenameOption opt) {
	//ordered insert to always keep the same filename layout
	name_options.insert(std::upper_bound(name_options.begin(), name_options.end(), opt), opt);
}

void log_context::add_entry(string name, string value) {
	data_insertion_order.insert(data_insertion_order.end(), name);
	data[name] = value;
}

void log_context::add_entry(string name, unsigned int value) {
	add_entry(name, to_string(value));
}

void log_context::add_entry(string name, size_t value) {
	add_entry(name, to_string(value));
}

void log_context::add_entry(string name, int value) {
	add_entry(name, to_string(value));
}

void log_context::add_title(string _title) {
	title = _title;
	add_filename_option(FilenameOption::TITLE);
}

//std::numeric_limits<float>::digits10 + 1

void log_context::add_entry(string name, float value) {
	//ensure max precision when converting to string
	std::ostringstream oss;
	oss << std::setprecision(std::numeric_limits<float>::digits10 + 1) << value;

	//add entry
	add_entry(name, oss.str());
}

void log_context::add_entry(const string name, double value) {
	std::ostringstream oss;
	oss << std::setprecision(std::numeric_limits<double>::digits10 + 1) << value;

	//add entry
	add_entry(name, oss.str());
}