#ifndef SPECTRAL_RT_PROJECT_FRAME_BUFFER_CUH
#define SPECTRAL_RT_PROJECT_FRAME_BUFFER_CUH

#include "vec3.cuh"

class frame_buffer {
public:
	frame_buffer(size_t _size) {
		size = _size;
		data = new vec3[byte_size()];
	}

	~frame_buffer() {
		delete[] data;
	}

	size_t byte_size() {
		return size * sizeof(vec3);
	}

	void split_channels(unsigned char* r, unsigned char* g, unsigned char* b) const {
		for (size_t i = 0; i < size; i++) {
			r[i] = static_cast<unsigned char>(data[i].e[0]);
			g[i] = static_cast<unsigned char>(data[i].e[1]);
			b[i] = static_cast<unsigned char>(data[i].e[2]);
		}
	}

	size_t size;
	vec3* data;
};

struct image_channels {

	image_channels(frame_buffer& fb) {
		size_t size = fb.size;
		r = new unsigned char[size];
		g = new unsigned char[size];
		b = new unsigned char[size];

		fb.split_channels(r, g, b);
	}
	image_channels& operator=(const frame_buffer& fb) {
		fb.split_channels(r, g, b);
		return *this;
	}


	~image_channels() {
		delete[] r;
		delete[] g;
		delete[] b;
	}

	unsigned char* r;
	unsigned char* g;
	unsigned char* b;
};

#endif