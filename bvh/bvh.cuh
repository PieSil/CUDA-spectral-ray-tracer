//
// Created by pietr on 12/12/2023.
//

#ifndef RTWEEKEND_CUDA_BVH_CUH
#define RTWEEKEND_CUDA_BVH_CUH

#include <algorithm>
#include "tri.cuh"
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

/*
class hittable_bbox {
    //a wrapper for the aabb class that allows to treat it as a hittable
public:

    __device__
    explicit hittable_bbox(const aabb &bbox) : bbox(bbox) {}

    __device__
    bool hit(const ray &r, float min, float max, hit_record &rec) const {
        return bbox.hit(r, min, max);
    }

    __host__ __device__
    aabb bounding_box() const {
        return bbox;
    }

private:
    aabb bbox;
};
 */

class bvh_node {

public:

    __device__
    explicit bvh_node(bool is_leaf) : is_leaf(is_leaf) {
        left = nullptr;
        right = nullptr;
    }

    __device__
    explicit bvh_node(tri* t) : primitive(t) {
        is_leaf = true;
        left = nullptr;
        right = nullptr;
    }

    __device__
    bvh_node(const bvh_node& other) : bbox(other.bbox), is_leaf(other.is_leaf) {
        //we want a shallow copy

        left = other.left;
        right = other.right;
        primitive = other.primitive;
    }

    __device__
    bvh_node& operator=(const bvh_node& r) = default;

    __device__
    bool hit(const ray &r, float min, float max, hit_record &rec) const;

    __host__ __device__
    aabb bounding_box() const {
        return is_leaf ? primitive->bounding_box() : bbox;
    }

    __device__
    void create_bbox() {
        if (left != nullptr && right != nullptr)
            bbox = aabb(left->bounding_box(), right->bounding_box());

        else if (left != nullptr) {
            bbox = aabb(left->bounding_box());
        }
        else if (right != nullptr) {
            bbox = aabb(right->bounding_box());
        }

    }

    __device__
    bvh_node* get_left() const {
        return left;
    }

    __device__
    bvh_node* get_right() const {
        return right;
    }

    __host__ __device__
    ~bvh_node() {
    }

    __host__ __device__
    void dealloc() {
        if (left != nullptr) {
            left->dealloc();
        }

        if (right != nullptr) {
            right->dealloc();
        }

        delete left;
        delete right;
    }

    bvh_node* left;
    bvh_node* right;
    bool is_leaf;
    tri* primitive;
    aabb bbox;

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

class bvh {
public:
    __device__
    bvh(tri** src_objects, size_t list_size, curandState* local_rand_state) {
        global_bbox = aabb(); //empty bbox
        if (list_size > 0 && build_bvh(src_objects, list_size, local_rand_state)) {
            build_nodes_bboxes();
            valid = true;
         } else {
            valid = false;
        }
    }

//    __device__
//    explicit bvh(const hittable** objects, const size_t n_objects, curandState* local_rand_state) : bvh(objects, n_objects, local_rand_state) {
//        //root = new bvh_node(list, local_rand_state);
//    }

    __host__ __device__
    ~bvh() {
        if (root != nullptr)
            root->dealloc();
        //delete[] nodes;
    }

    __device__
    bool build_bvh(tri** src_objects, size_t list_size, curandState* local_rand_state);

    __device__
    void build_nodes_bboxes();

    static __device__
    bvh_node* get_left_child(bvh_node* node) {
        return node->get_left();
    }

    static __device__
    bvh_node* get_right_child(bvh_node* node) {
        return node->get_right();
    }

    __device__
    bool hit(const ray &r, float min, float max, hit_record &rec) const;

    __device__
    static bool hit(const ray& r, float min, float max, hit_record& rec, bvh_node* root, bool is_valid);

    __host__ __device__
    aabb bounding_box() const {
        return global_bbox;
    }

    __device__
        void to_shared(bvh_node* shared_mem, const size_t& shared_mem_size) const;

    __device__
    static bool box_compare(
    const tri* a, const tri* b, int axis_index) {
       // return a->bounding_box().center()[axis_index] < b->bounding_box().center()[axis_index];
            return a->bounding_box().axis(axis_index).min < b->bounding_box().axis(axis_index).min;
    }

    __device__
    static bool box_x_compare (const tri* a, const tri* b) {
        return box_compare(a, b, 0);
    }

    __device__
    static bool box_y_compare (const tri* a, const tri* b) {
        return box_compare(a, b, 1);
    }

    __device__
    static bool box_z_compare (const tri* a, const tri* b) {
        return box_compare(a, b, 2);
    }

    __device__
    bool is_valid() const {
        return valid;
    }

    __device__
    bvh_node* getRoot() const {
        return root;
    };

private:
    //bvh_node** nodes;
    uint size;
    bvh_node* root;
    aabb global_bbox;
    bool valid;

};


#endif //RTWEEKEND_CUDA_BVH_CUH
