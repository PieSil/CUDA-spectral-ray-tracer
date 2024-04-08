//
// Created by pietr on 20/03/2024.
//

#ifndef SPECTRAL_RAY_TRACING_VEC3_H
#define SPECTRAL_RAY_TRACING_VEC3_H

#include "host_utility.cuh"
#include "cuda_utility.cuh"

class vec3 {
public:
    float e[3];

    __host__ __device__
    vec3() : e{.0f, .0f, .0f} {};

    __host__ __device__
    vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __host__ __device__
    float x() const { return e[0]; }
    __host__ __device__
    float y() const { return e[1]; }
    __host__ __device__
    float z() const { return e[2]; }

    __host__ __device__
    vec3 operator-() const { return vec3(-e[0], e[1], e[2]); }

    __host__ __device__
    float operator[](int i) const { return e[i]; }

    __host__ __device__
    float& operator[](int i) { return e[i]; }

    __host__ __device__
    vec3& operator+=(const vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__
    vec3& operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__
    vec3& operator/=(float t) {
        return *this *= 1/t;
    }


    __host__ __device__
    float length_squared() const {
        float x = e[0];
        float y = e[1];
        float z = e[2];
        return x*x + y*y + z*z;
    }


    __host__ __device__
    float length() const {
        return sqrt(length_squared());
    }

    __host__ __device__
    static vec3 matrix_mul(vec3 vec, const float* linearized_matrix) {
        float e0 = vec.x();
        float e1 = vec.y();
        float e2 = vec.z();

        vec3 res = vec3((linearized_matrix[0] * e0) + (linearized_matrix[1] * e1) + (linearized_matrix[2] * e2),
                        (linearized_matrix[3] * e0) + (linearized_matrix[4] * e1) + (linearized_matrix[5] * e2),
                        (linearized_matrix[6] * e0) + (linearized_matrix[7] * e1) + (linearized_matrix[8] * e2));

        return res;
    }

    __host__ __device__
    bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        auto s = 1e-8f;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    //static methods
    __device__
    static vec3 random(curandState* local_rand_state) {
        return vec3(cuda_random_float(local_rand_state), cuda_random_float(local_rand_state), cuda_random_float(local_rand_state));
    }

    __device__
    static vec3 random(float min, float max, curandState* local_rand_state) {
        return vec3(cuda_random_float(min,max, local_rand_state), cuda_random_float(min,max, local_rand_state), cuda_random_float(min,max, local_rand_state));
    }
};

__host__
inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v[0] << ' ' << v[1] << ' ' << v[2];
}

__host__ __device__
inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

__host__ __device__
inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

__host__ __device__
inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__
inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.x(), t*v.y(), t*v.z());
}

__host__ __device__
inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

__host__ __device__
inline vec3 operator/(vec3 v, float t) {
    return (1/t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
           + u.e[1] * v.e[1]
           + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__host__ __device__
inline vec3 reflect(const vec3&v, const vec3&n) {
    /*
     * Compute reflection of vector v given a surface with normal n
     * dot(v, n): gets the (scalar) component of v along the direction of n
     * dot(v, n) * n: gets the vector of length dot(v,n) along the direction of n
     * v-dot(v, n)*n: negates the component of v parallel to the direction of n
     * v-2*dot(v, n)*n: reflects v wrt the direction of n
     */

    return v - 2*dot(v,n)*n;
}

__host__ __device__
inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {

    /*
     * For the full math please refer to to chapter 11.2 of:
     * Ray Tracing in One Weekend by  Peter Shirley, Trevor David Black, Steve Hollasch
     * (https://raytracing.github.io/books/RayTracingInOneWeekend.html)
     */

    auto cos_theta = fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}


//functions that involve random numbers (device only)
__device__
inline vec3 random_in_unit_sphere(curandState* local_rand_state) {
    while (true) {
        /* Generate a random vector inside the unit cube and reject it until it falls into the unit sphere */
        //TODO: can this be optimized?
        auto p = vec3::random(-1,1, local_rand_state);
        if (p.length_squared() < 1.0f)
            return p;
    }
}

__device__
inline vec3 random_unit_vector(curandState* local_rand_state) {
    /*
    * Generate random vector in the unit sphere and normalize it
    * (so that it's ON (not in) the unit sphere)
    */
    return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__
inline vec3 random_on_hemisphere(const vec3& normal, curandState* local_rand_state) {
    vec3 on_unit_sphere = random_unit_vector(local_rand_state);
    if (dot(on_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        /*  Flip the unit vector */
        return -on_unit_sphere;
}

__device__
inline vec3 random_in_unit_disk(curandState* local_rand_state) {
    while (true) {
        auto p = vec3(cuda_random_float(-1,1, local_rand_state), cuda_random_float(-1,1, local_rand_state), 0);
        if (p.length_squared() < 1.0f)
            return p;
    }
}

__device__
inline float distance(vec3 start, vec3 end) {
    float xDiff = start.x() - end.x();
    float xSqr = xDiff * xDiff;

    float yDiff = start.y() - end.y();
    float ySqr = yDiff * yDiff;

    float zDiff = start.z() - end.z();
    float zSqr = zDiff * zDiff;

    float distSqr = xSqr + ySqr + zSqr;
    return sqrt(distSqr);
}




#endif //SPECTRAL_RAY_TRACING_VEC3_H
