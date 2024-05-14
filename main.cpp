#include <iostream>
#include "scene.cuh"
#include "device_init.cuh"
#include "save_image.h"
#include "image.h"
#include "render_manager.cuh"

#define SAMPLES_PER_PIXEL 100
#define BOUNCE_LIMIT 10

using namespace scene;

bool render_loop(render_manager& rm, frame_buffer& fb, uchar_img& dst_image, img_display& disp, image_channels& ch) {
    bool completed = false;

    if (rm.isReadyToRender()) {
        disp.display(dst_image);

        clock_t start, stop;
        start = clock();

        std::clog << "Rendering... ";

        while (!rm.isDone()) {
            rm.step();
            rm.update_fb();
            ch = fb;
            dst_image = get_image(ch.r, ch.g, ch.b, rm.getImWidth(), rm.getImHeight());
            disp.render(dst_image);
            disp.paint();
        }

        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::clog << "done, took " << timer_seconds << " seconds.\n";
        completed = true;
    }
    else {
        cerr << "Device parameters not yet initialized";
    }

    return completed;
}

int main() {
    init_device_symbols();
    
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

        render_manager rm(sm.getWorld(), sm.getCamPtr(), fb.data);

        rm.init_renderer(BOUNCE_LIMIT, SAMPLES_PER_PIXEL);
        rm.init_device_params(100, 100);

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
       
        if (render_loop(rm, fb, image, disp, ch)) {
            save_img(image, "test.bmp");
            while (!disp.is_closed()) {
                disp.wait();
            }
        }
       
        //write_to_ppm(fb.data, width, height);
    } else {
        cerr << res.msg << endl;
    }
    return 0;
}


