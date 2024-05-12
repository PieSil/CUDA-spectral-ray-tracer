#include <iostream>
#include "scene.cuh"
#include "device_init.cuh"
#include "save_image.h"

#define SAMPLES_PER_PIXEL 500
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
        sm.render(&fb, BOUNCE_LIMIT, SAMPLES_PER_PIXEL);
        
        image_channels ch(fb);
        save_img(ch.r, ch.g, ch.b, width, height, "test.bmp");
        

        //write_to_ppm(fb.data, width, height);
    } else {
        cerr << res.msg << endl;
    }
    return 0;
}
