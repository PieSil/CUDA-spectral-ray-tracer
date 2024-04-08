//
// Created by pietr on 12/12/2023.
//

#ifndef RTWEEKEND_CUDA_BVH_CUH
#define RTWEEKEND_CUDA_BVH_CUH

#include <algorithm>
#include "hittable.cuh"
#include "cuda_utility.cuh"

#define MAX_DEPTH 64

/*
 * bvh build logic and traversal inspired from:
 * https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
 * https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
 *
 * and adapted to the "Ray Tracing: The Next Week" book way of building the bvh (chapters 3.7 to 3.9):
 * https://raytracing.github.io/books/RayTracingTheNextWeek.html#boundingvolumehierarchies/thebvhnodeclass
 */


class hittable_bbox : public hittable {
    //a wrapper for the aabb class that allows to treat it as a hittable
public:

    __device__
    explicit hittable_bbox(const aabb &bbox) : bbox(bbox) {}

    __device__
    bool hit(const ray &r, float min, float max, hit_record &rec) const override {
        return bbox.hit(r, min, max);
    }

    __host__ __device__
    aabb bounding_box() const override {
        return bbox;
    }

private:
    aabb bbox;
};

class bvh_node : public hittable {

public:

    __device__
    explicit bvh_node(bool is_leaf) : is_leaf(is_leaf) {
        left = nullptr;
        right = nullptr;
        hit_volume = nullptr;
    }

    __device__
    explicit bvh_node(hittable* h) : hit_volume(h) {
        is_leaf = true;
        left = nullptr;
        right = nullptr;
    }

    __device__
    bool hit(const ray &r, float min, float max, hit_record &rec) const override;

    __host__ __device__
    aabb bounding_box() const override {
        return hit_volume->bounding_box();
    }

    __device__
    void create_bbox() {
        if (left != nullptr && right != nullptr)
            hit_volume = new hittable_bbox(aabb(left->bounding_box(), right->bounding_box()));

        else if (left != nullptr) {
            hit_volume = new hittable_bbox(left->bounding_box());
        }
        else if (right != nullptr) {
            hit_volume = new hittable_bbox(right->bounding_box());
        }

    }

    __host__ __device__
    ~bvh_node() override {
        if (!is_leaf)
            delete hit_volume;

        delete left;
        delete right;
    }

    bvh_node* left;
    bvh_node* right;
    bool is_leaf;
    hittable* hit_volume;

};

struct stack_item {
    __device__
    stack_item(size_t s, size_t e) : start(s), end(e) {

    }

    __device__
    stack_item() {};

    size_t start;
    size_t end;
};

class bvh : public hittable {
public:
    __device__
    bvh(hittable** src_objects, size_t list_size, curandState* local_rand_state) {
        global_bbox = aabb(); //empty bbox
        if (list_size > 0 && build_bvh(src_objects, list_size, local_rand_state)) {
            build_nodes_bboxes();
            valid = true;
           /* size_t byte_size = sizeof(hittable*)*list_size + sizeof(hittable)*list_size + sizeof(bvh_node)*(list_size-1);
            printf("At least %llu bytes\n", byte_size);*/
         } else {
            valid = false;
        }
    }

//    __device__
//    explicit bvh(const hittable** objects, const size_t n_objects, curandState* local_rand_state) : bvh(objects, n_objects, local_rand_state) {
//        //root = new bvh_node(list, local_rand_state);
//    }

    __host__ __device__
    ~bvh() override {
        delete root;
        //delete[] nodes;
    }

    __device__
    bool build_bvh(hittable** src_objects, size_t list_size, curandState* local_rand_state);

    __device__
    void build_nodes_bboxes();



    static __device__
    bvh_node* get_left_child(bvh_node* node) {
        return node->left;
    }

    static __device__
    bvh_node* get_right_child(bvh_node* node) {
        return node->right;
    }

    __device__
    bool hit(const ray &r, float min, float max, hit_record &rec) const override;

    __host__ __device__
    aabb bounding_box() const override {
        return global_bbox;
    }

    __device__
    static bool box_compare(
            const hittable* a, const hittable* b, int axis_index) {
        return a->bounding_box().axis(axis_index).min < b->bounding_box().axis(axis_index).min;
    }

    __device__
    static bool box_x_compare (const hittable* a, const hittable* b) {
        return box_compare(a, b, 0);
    }

    __device__
    static bool box_y_compare (const hittable* a, const hittable* b) {
        return box_compare(a, b, 1);
    }

    __device__
    static bool box_z_compare (const hittable* a, const hittable* b) {
        return box_compare(a, b, 2);
    }

    __device__
    bool is_valid() const {
        return valid;
    }

private:
    //bvh_node** nodes;
    uint size;
    bvh_node* root;
    aabb global_bbox;
    bool valid;

};


#endif //RTWEEKEND_CUDA_BVH_CUH
