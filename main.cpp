#include <iostream>
#include "scene.cuh"
#include "device_init.cuh"
#include "save_image.h"
#include "image.h"
#include "render_manager.cuh"

#define SAMPLES_PER_PIXEL 100
#define BOUNCE_LIMIT 10

using namespace scene;

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
        disp.display(image);

        render_manager rm(sm.getWorld(), sm.getCamPtr(), fb.data);

        rm.init_renderer(BOUNCE_LIMIT, SAMPLES_PER_PIXEL);
        rm.init_device_params(100, 100);

        clock_t start, stop;
        start = clock();

        std::clog << "Rendering... ";

        while (!rm.isDone()) {
            rm.step();
            rm.update_fb();
            ch = fb;
            image = get_image(ch.r, ch.g, ch.b, width, height);
            disp.render(image);
            disp.paint();
        }


        stop = clock();
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::clog << "done, took " << timer_seconds << " seconds.\n";
        
        /*
        ch = fb;
        image = get_image(ch.r, ch.g, ch.b, width, height);
        disp.render(image);
        disp.paint();
        */
        
        save_img(image, "test.bmp");
        while (!disp.is_closed()) {
            disp.wait();
        }
        

        //write_to_ppm(fb.data, width, height);
    } else {
        cerr << res.msg << endl;
    }
    return 0;
}
