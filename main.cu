#include <iostream>
#include "scene.cuh"
#include "device_init.cuh"

#define SAMPLES_PER_PIXEL 500
#define BOUNCE_LIMIT 10

using namespace scene;

int main() {
    init_device_symbols();

    scene_manager sm = scene_manager();
    result res = sm.getResult();
    if (res.success) {
        clog << res.msg << endl;
        sm.render(BOUNCE_LIMIT, SAMPLES_PER_PIXEL);
    } else {
        cerr << res.msg << endl;
    }
    return 0;
}
