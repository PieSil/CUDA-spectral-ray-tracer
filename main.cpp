#include <iostream>
#include "scene.cuh"
#include "device_init.cuh"
#include "save_image.h"
#include "image.h"
#include "render_manager.cuh"
#include "log_context.h"
#include "params.h"
#include <cuda_profiler_api.h> 

#define SAMPLES_PER_PIXEL 100
#define BOUNCE_LIMIT 10

using namespace scene;

bool render_cycle(render_manager& rm, frame_buffer& fb, uchar_img& dst_image, img_display& disp, image_channels& ch, bool multithread = true) {
	bool completed = false;
	if (rm.isReadyToRender()) {
		auto pm = param_manager::getInstance();
		if (pm->getParams().showRender())
			disp.display(dst_image);

		clock_t start, stop;
		start = clock();

		std::clog << "Rendering... ";

		cudaProfilerStart();
		if (multithread) {
			rm.render_cycle();
			bool has_data = true;

			do {
				has_data = rm.update_fb();
				ch = fb;
				dst_image = get_image(ch.r, ch.g, ch.b, rm.getImWidth(), rm.getImHeight());
				if (pm->getParams().showRender()) {
					disp.render(dst_image);
					disp.paint();
				}
			} while (has_data);

		}
		else {
			bool has_data = true;
			do {
				has_data = rm.step();
				rm.update_fb();
				ch = fb;
				dst_image = get_image(ch.r, ch.g, ch.b, rm.getImWidth(), rm.getImHeight());
				if (pm->getParams().showRender()) {
					disp.render(dst_image);
					disp.paint();
				}
			} while (has_data);
		}
		cudaProfilerStop();

		stop = clock();
		double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
		auto lc = log_context::getInstance();
		lc->add_entry("total rendering time (seconds)", timer_seconds);
		std::clog << "done, took " << timer_seconds << " seconds.\n";
		rm.end_render();
		completed = true;
	}
	else {
		cerr << "Device parameters not yet initialized";
	}

	return completed;
}

void render(bool multithread = true) {
	scene_manager sm = scene_manager();
	result res = sm.getResult();
	if (res.success) {
		clog << res.msg << endl;
		uint width = sm.img_width();
		uint height = sm.img_height();
		frame_buffer fb(width * height);
		image_channels ch(fb);
		uchar_img image = get_image(ch.r, ch.g, ch.b, width, height);
		img_display disp = get_image_display("Render");

		render_manager rm(sm.getWorld(), sm.getMaterials(), sm.getCamPtr(), &fb);

		auto pm = param_manager::getInstance();
		rm.init_renderer(pm->getParams().getBounceLimit(), pm->getParams().getNSamples());
		rm.init_device_params(pm->getParams().getXcsize(), pm->getParams().getYcsize());

		/*
		while (!rm.isDone()) {
			rm.step();
			rm.update_fb();
			ch = fb;
			image = get_image(ch.r, ch.g, ch.b, width, height);
			disp.render(image);
			disp.paint();
		}
		*/

		/*
		ch = fb;
		image = get_image(ch.r, ch.g, ch.b, width, height);
		disp.render(image);
		disp.paint();
		*/
		if (render_cycle(rm, fb, image, disp, ch, multithread)) {
			if (param_manager::getInstance()->getParams().logActive())
				log_context::getInstance()->to_file();

			string ext = ".bmp";
			string image_filename = param_manager::getInstance()->getParams().getImgTitle() + ext;
			string_to_filename(image_filename);

			if (pm->getParams().doSaveImage())
				save_img(image, image_filename.c_str());

			if (pm->getParams().showRender()) {
				while (!disp.is_closed()) {
					disp.wait();
				}
			}

		}

		//write_to_ppm(fb.data, width, height);
	}
	else {
		cerr << res.msg << endl;
	}
}

int main(int argc, char* argv[]) {
	auto pm = param_manager::getInstance();
	pm->parseArgs(argc, argv);
	init_device_symbols();
	auto lc = log_context::getInstance();
	cout << "Image Title: " << pm->getParams().getImgTitle() << endl;

	string log_subdir = pm->getParams().getLogSubdir();

	if (!log_subdir.empty()) {
		cout << "Log Subdir: " << log_subdir << endl;
	}

	const uint scene_id = pm->getParams().getSceneId();
	cout << "Scene: " << sceneIdToStr[scene_id] << " (ID: " << scene_id << ")" << endl;
	cout << "X res: " << pm->getParams().getXres() << endl;
	cout << "Y res: " << pm->getParams().getYres() << endl;
	cout << "AR: " << pm->getParams().getAR() << endl;
	cout << "X chunk size: " << pm->getParams().getXcsize() << endl;
	cout << "Y chunk size: " << pm->getParams().getYcsize() << endl;
	cout << "# samples: " << pm->getParams().getNSamples() << endl;
	cout << "# max bounces: " << pm->getParams().getBounceLimit() << endl;
	const bool do_log = pm->getParams().logActive();
	cout << "Logging " << (do_log ? "enabled" : "disabled") << endl;

	lc->append_dir(pm->getParams().getLogSubdir());
	lc->add_title(pm->getParams().getImgTitle());
	lc->add_filename_option(FilenameOption::TIMESTAMP);

	render();

	return 0;
}


