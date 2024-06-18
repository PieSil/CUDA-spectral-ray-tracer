#ifndef SPECTRAL_RT_PROJECT_FRAME_BUFFER_CUH
#define SPECTRAL_RT_PROJECT_FRAME_BUFFER_CUH

#include "vec3.cuh"

class frame_buffer {
public:
	frame_buffer(size_t img_size) {
		channel_size = img_size;
		r = new float[single_channel_byte_size()];
		g = new float[single_channel_byte_size()];
		b = new float[single_channel_byte_size()];
	}

	~frame_buffer() {
		delete[] r;
		delete[] g;
		delete[] b;
	}

	const size_t single_channel_byte_size() const {
		return channel_size * sizeof(float);
	}

	void split_channels(unsigned char* const _r, unsigned char* const _g, unsigned char* const _b) const {
		for (size_t i = 0; i < channel_size; i++) {
			_r[i] = static_cast<unsigned char>(r[i]);
			_g[i] = static_cast<unsigned char>(g[i]);
			_b[i] = static_cast<unsigned char>(b[i]);
		}
	}

	size_t channel_size;
	float* r;
	float* g;
	float* b;
};

struct image_channels {

	image_channels(frame_buffer& fb) {
		size_t channel_size = fb.channel_size;
		r = new unsigned char[channel_size];
		g = new unsigned char[channel_size];
		b = new unsigned char[channel_size];

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