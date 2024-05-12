//
// Created by pietr on 13/04/2024.
//

#include "camera.cuh"

__host__ void camera::initialize() {
    /* Calculate the image height, and ensure that it's at least 1 */
    image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    num_pixels = image_width * image_height;

    center = lookfrom;

    /* Determine viewport dimensions */
    //auto focal_length = (lookfrom - lookat).length();
    auto theta = degrees_to_radians(vfov); //vertical angle covered by the camera view
    auto h = tan(theta/2.0f) * focus_dist; //h = half height of the image plane
    auto viewport_height = 2.0f * h;

    /*
     * Viewport widths less than one are ok since they are real valued
     * In order to compute the viewport height we do NOT use the value of aspect_ratio
     * since it only defines our ideal target ratio.
     * We compute the effective aspect ratio as image_width/image_height.
     */
    auto viewport_width = viewport_height * (static_cast<float>((float)image_width/(float)image_height));

    // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
    vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u = viewport_u / image_width;
    pixel_delta_v = viewport_v / image_height;

    /*
    * Calculate the location of the upper left pixel
    * NOTE: the camera coordinate system has the camera facing the negative z-axis,,
    * the focal length is defined by the distance between the camera center and the viewport.
    * Thus in order to compute viewport position on the z axis we need to subtract
    * a vector parallel to the camera facing direction with a magnitude of focal_length.
    */
    auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;

    /* Pixel grid is inset by half the pixel-to-pixel distance wrt viewport edges */
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v); //upper left pixel center coordinates

    // Calculate the camera defocus disk basis vectors.
    auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;
}
