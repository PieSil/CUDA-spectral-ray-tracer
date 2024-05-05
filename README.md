# A CUDA spectral ray tracer

## Intro
This is a project I'm currently developing for the "GPU Programming" course at the "Politecnico di Torino", it consists of a basic spectral ray tracer able to render some relatively contained hard-coded scenes.
Since the aim of this project is mainly to practice CUDA language and GPU code optimization the physics related aspects are secondary, and developing a physically correct behaviour is not a priority, it is also worth noting that my efforts are dedicated to the rendering algorithm, while the other components of this project (e.g. color conversion, scene building and definition) are merely needed for the rendering algorithm to work and, as I'm writing this, I have no plan to further optimize or smooth out these aspects.

__Please note that this is still a work in progress, as such the code may appear particularly rough, unorganized and poorly documented, I also do not guarantee that at any given point in time during development this project will compile and/or run as expected.__

## Details and references
It currently supports lambertian, reflective and dielectric materials, and the spectral behaviour is obtained by following the "hero wavelength" approach proposed in *A. Wilkie, S. Nawaz, M. Droske, A. Weidlich, and J. Hanika. 2014. Hero wavelength spectral sampling. In Proceedings of the 25th Eurographics Symposium on Rendering (EGSR '14). Eurographics Association, Goslar, DEU, 123–131. https://doi.org/10.1111/cgf.12419*.

The implementation is based on the first two books of the _Ray Tracing in One Weekend_ series by _Peter Shirley, Trevor David Black and Steve Hollasch (https://raytracing.github.io/)_.  
I also found myself to face the problem of converting an RGB color to an emission or reflectance spectrum which, under the d65 illuminant, represents said color, in order to do that I slightly adapted the code proposed in the _Physically Based Ray Tracing_ repository (https://github.com/mmp/pbrt-v4) which follows the approach described in the PBRT book (_https://www.pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Color_).