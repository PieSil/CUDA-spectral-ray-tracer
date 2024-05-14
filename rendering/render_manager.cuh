#ifndef SPECTRAL_RT_PROJECT_RENDER_MANAGER_CUH
#define SPECTRAL_RT_PROJECT_RENDER_MANAGER_CUH

#include "camera.cuh"
#include "rendering.cuh"

class render_manager {
public:

    render_manager(bvh** _dev_bvh, camera* _cam, vec3* _fb) {
        if (_dev_bvh != nullptr && _cam != nullptr && _fb != nullptr) {
            dev_bvh = _dev_bvh;
            cam = _cam;
            fb = _fb;
            image_width = cam->getImageWidth();
            image_height = cam->getImageHeight();
            scene_inited = true;
        }
    }

    ~render_manager() {
        if (device_inited) {
            delete[] tmp_fb;
        }
    }

    void step();

    void init_renderer(uint bounce_limit, uint samples_per_pixel);

    void init_device_params(dim3 threads, dim3 blocks, uint _chunk_width, uint _chunk_height);

    void init_device_params(uint _chunk_width, uint _chunk_height);

    void init_device_params();

    void update_fb() {
        
        if (device_inited) {
            for (size_t row = 0; row < last_chunk_height; row++) {
                for (size_t col = 0; col < last_chunk_width; col++) {
                    size_t tmp_pixel_index = row * last_chunk_width + col;
                    size_t fb_pixel_index = (row + last_offset_y) * cam->getImageWidth() + (col + last_offset_x);
                    fb[fb_pixel_index] = tmp_fb[tmp_pixel_index];
                }
            }
        }
        
    }

    const vec3* getFB() const {
        return tmp_fb;
    }

    const bool isDone() const {
        return i >= n_iterations;
    }

private:

    bvh** dev_bvh;
    camera* cam;
    uint image_width;
    uint image_height;
    bool scene_inited = false;

	renderer r;
	bool renderer_inited = false;
    vec3* fb;
    vec3* tmp_fb;

    bool device_inited = false;

    uint i = 0;
    uint chunk_width;
    uint chunk_height;
    uint n_iterations;
    //uint chunk_size;
    uint x_chunks;
    //uint y_chunks;
    uint offset_x = 0;
    uint offset_y = 0;
    uint last_offset_x = 0;
    uint last_offset_y = 0;
    uint last_chunk_width = 0;
    uint last_chunk_height = 0;
};

#endif