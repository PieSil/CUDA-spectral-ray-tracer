#include "transform.cuh"

__host__ __device__
void transform::assign_rot_matrix(float theta, AXIS ax, float rot_matrix[9]) {

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    switch (ax) {
        case X:
            rot_matrix[4] = cos_theta;
            rot_matrix[5] = -sin_theta;
            rot_matrix[7] = sin_theta;
            rot_matrix[8] = cos_theta;
            break;

        case Y:
            rot_matrix[0] = cos_theta;
            rot_matrix[2] = sin_theta;
            rot_matrix[6] = -sin_theta;
            rot_matrix[8] = cos_theta;
            break;

        case Z:
            rot_matrix[0] = cos_theta;
            rot_matrix[1] = -sin_theta;
            rot_matrix[3] = sin_theta;
            rot_matrix[4] = cos_theta;
            break;

        default:
            break;
    }
}
