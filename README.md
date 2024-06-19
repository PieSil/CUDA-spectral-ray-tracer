# A CUDA spectral ray tracer

## Intro
This is a project I'm currently developing for the "GPU Programming" course at the "Politecnico di Torino", it consists of a basic spectral ray tracer able to render some relatively simple hard-coded scenes.
Since the aim of this project is mainly to practice CUDA language and GPU code optimization the physics related aspects are secondary, it is also worth noting that my efforts are dedicated to the rendering algorithm, while the other components of this project (e.g. color conversion, scene building and definition) are merely needed for the rendering algorithm to work and, as I'm writing this, I have no plan to further optimize or smooth out these aspects.

The only color space I'm supporting is sRGB since I'm trying to keep things as simple as possible (but I do use CIE XYZ in order to extract color info from spectral data).

## Details and references
It currently supports lambertian, reflective and dielectric materials, and the spectral behaviour is obtained by following the "hero wavelength" approach proposed in *A. Wilkie, S. Nawaz, M. Droske, A. Weidlich, and J. Hanika. 2014. Hero wavelength spectral sampling. In Proceedings of the 25th Eurographics Symposium on Rendering (EGSR '14). Eurographics Association, Goslar, DEU, 123–131. https://doi.org/10.1111/cgf.12419*.

The implementation is based on the first two books of the _Ray Tracing in One Weekend_ series by _Peter Shirley, Trevor David Black and Steve Hollasch (https://raytracing.github.io/)_.  
I also found myself to face the problem of converting an RGB color to an emission or reflectance spectrum which, under the d65 illuminant, represents said color, in order to do that I slightly adapted the code proposed in the _Physically Based Ray Tracing_ repository (https://github.com/mmp/pbrt-v4) which follows the approach described in the PBRT book (_https://www.pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Color_).

The BVH was implemented by merging the behaviour described in chapter 3 of the second book from the _Ray Tracing in One Weekend_ series (_Ray Tracing: The Next Week https://raytracing.github.io/books/RayTracingTheNextWeek.html_) with the traversal and construction techniques described by Tero Karras in his _"Thinking parallel"_ articles on the official nVidia blog (part II and III specifically: https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/ https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)
