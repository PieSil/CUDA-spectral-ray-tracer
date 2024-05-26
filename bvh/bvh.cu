//
// Created by pietr on 12/12/2023.
//

#include "bvh.cuh"

__device__
void swap(tri** src_objects, int a, int b) {
    auto t = src_objects[a];
    src_objects[a] = src_objects[b];
    src_objects[b] = t;
}

__device__
int partition(tri** src_objects, int l, int h, bool(*compare)(const tri*, const tri*)) {
    if(l == h)
        return l;

    tri* x = src_objects[h];
    int i = (l - 1);

    for (int j = l; j < h; j++) {
        if (compare(src_objects[j], x)) {
            i++;
            swap(src_objects, i, j);
        }
    }

    swap(src_objects, i + 1, h);
    return (i + 1);
}

__device__
void quicksort_primitives(tri** src_objects, int start, int end, bool(*compare)(const tri*, const tri*)) {
    // Create an auxiliary stack
    auto stack = new int[end - start + 1];

    // initialize top of stack
    int top = -1;

    // push initial values of l and h to stack
    stack[++top] = start;
    stack[++top] = end;

    // Keep popping from stack while is not empty
    while (top >= 0) {
        // Pop h and l
        end = stack[top--];
        start = stack[top--];

        // Set pivot element at its correct position
        // in sorted array
        int p = partition(src_objects, start, end, compare);

        // If there are elements on left side of pivot,
        // then push left side to stack
        if (p - 1 > start) {
            stack[++top] = start;
            stack[++top] = p - 1;
        }

        // If there are elements on right side of pivot,
        // then push right side to stack
        if (p + 1 < end) {
            stack[++top] = p + 1;
            stack[++top] = end;
        }
    }

    delete[] stack;
}

__device__
bool bvh_node::hit(const ray &r, float min, float max, hit_record &rec) const {

    return is_leaf ? primitive->hit(r, min, max, rec) : bbox.hit(r, min, max);

    /*if(!hit_volume->hit(r, ray_t, rec))
        return false;

    bool hit_left = left != nullptr && left->hit(r, ray_t, rec);
    bool hit_right = right != nullptr && right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

    return hit_left || hit_right;*/
}


/*
__device__
bool bvh::hit(const ray &r, float min, float max, hit_record &rec) const {
    bool res = false;

    if (is_valid()) {
        res = hit(r, min, max, rec, root);
    }
    return res;
}
*/

__device__
bool bvh::hit(const ray &r, float min, float max, hit_record &rec, bvh_node* shared_mem_nodes, uint shared_mem_size) {

    bool hit_anything = false;
    float closest_so_far = max;
    //printf("computing hits\n");

    // Allocate traversal stack from thread-local memory,
    // and push NULL to indicate that there are no postponed nodes.
    bvh_node* stack[64];
    bvh_node** stack_ptr = stack;
    *stack_ptr++ = nullptr; // push

    // Traverse nodes starting from the root.
    bvh_node* node = (0 < shared_mem_size) ? &shared_mem_nodes[0] : &nodes[0]; 

    if (node->is_leaf) {
        //only one element
        if (node->hit(r, min, closest_so_far, rec)) {
            hit_anything = true;
            closest_so_far = rec.t;
        }
    } else do {
            //bvh_node* child_l = node->get_left(block_mutex, node_cache, cur_cache_idx, cache_size);
            //bvh_node* child_r = node->get_right(block_mutex, node_cache, cur_cache_idx, cache_size);

        //access shared memory when possible
        uint idx = node->left_idx;
        bvh_node* child_l = idx == -1 ? nullptr : ((idx < shared_mem_size) ? &shared_mem_nodes[idx] : &nodes[idx]);//node->get_left();

        idx = node->right_idx;
        bvh_node* child_r = idx == -1 ? nullptr : ((idx < shared_mem_size) ? &shared_mem_nodes[idx] : &nodes[idx]);;//node->get_right();

            hit_record temp_rec;
            bool hits_l = child_l != nullptr && (child_l->hit(r, min, closest_so_far, temp_rec));
            if (hits_l && child_l->is_leaf) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }

            bool hits_r = child_r != nullptr && (child_r->hit(r, min, closest_so_far, temp_rec));
            if (hits_r && child_r->is_leaf) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }


            /*// Query overlaps a leaf node => report collision.
            if (overlapL && bvh.isLeaf(childL))
                list.add(queryObjectIdx, bvh.getObjectIdx(childL));

            if (overlapR && bvh.isLeaf(childR))
                list.add(queryObjectIdx, bvh.getObjectIdx(childR));*/

            // Query overlaps an internal node => traverse.
            bool traverse_l = child_l != nullptr && (hits_l && !child_l->is_leaf);
            bool traverse_r = child_r != nullptr && (hits_r && !child_r->is_leaf);

            if (!traverse_l && !traverse_r)
                node = *--stack_ptr; // pop
            else {
                node = traverse_l ? child_l : child_r;
                if (traverse_l && traverse_r)
                    *stack_ptr++ = child_r; //push
            }

        } while (node != nullptr);

    return hit_anything;

}

//__device__
//void bvh::to_shared(bvh_node* shared_mem, const size_t& shared_mem_size) const {
//    //breadth first traverse of bvh to copy higher nodes into shared memory
//    shared_mem[0] = *root;
//    size_t queue_start = 0;
//    size_t queue_end = 1;
//
//    //iterate over node queue, break loop if shared memory is full
//    while (queue_end < shared_mem_size && queue_start < queue_end) {
//        bvh_node* current = &shared_mem[queue_start];
//        
//        //check if left node exists
//        if (current->left != nullptr) {
//            //move node to shared memory
//            shared_mem[queue_end] = *(current->left);
//            //update pointer to child of current node
//            current->left = &shared_mem[queue_end];
//            //printf("copied left node to shared memory at index %d\n", queue_end);
//
//            queue_end++;
//        }
//
//        //check if right node exists, check for out of bound access in case shared memory is full
//        if (queue_end < shared_mem_size && current->right != nullptr) {
//            //move node to shared memory
//            shared_mem[queue_end] = *(current->right);
//            //update pointer to child of current node
//            current->right = &shared_mem[queue_end];
//            //printf("copied left node to shared memory at index %d\n", queue_end);
//            queue_end++;
//        }
//
//        //process next node
//        queue_start++;
//    }
//}

__device__ bool bvh::build_bvh(tri** src_objects, size_t list_size, curandState* local_rand_state) {

    //dynamic allocation, I do not care about performance during BVH construction anyway since it is only built once
    auto stack = new stack_item[MAX_DEPTH];
    auto node_stack = new  bvh_node*[MAX_DEPTH];

    int tos = -1;
    int n_nodes = 0;
    uint cur_idx = 0;
    //int stack_tos = -1;
    //int nodes_tos = -1;
    nodes[cur_idx] = bvh_node(false); //root
    root_idx = 0;
    //root = new bvh_node(false);
    tos++;
    n_nodes++;
    stack[tos] = stack_item(0, list_size);
    node_stack[tos] = &nodes[cur_idx];
    cur_idx++;

    while(tos >= 0) {
        //pop
        stack_item current = stack[tos];
        bvh_node* node = node_stack[tos];
        tos--;

        size_t current_span = current.end - current.start;
        if (current_span > 0) {
            bool is_leaf = (current_span == 1);

            if (is_leaf) {
                //create leaf node
                node->is_leaf = true;
                node->left_idx = -1;
                //node->left = nullptr;
                node->right_idx = -1;
                //node->right = nullptr;
                node->primitive = src_objects[current.start];
            } else {
                int axis = cuda_random_int(0, 2, local_rand_state);
                auto comparator = (axis == 0) ? box_x_compare: ((axis == 1) ? box_y_compare: box_z_compare);
                //auto comparator = box_x_compare;

                if (current_span == 2) {
                    //create left and right leaves
                    if (comparator(src_objects[current.start], src_objects[current.start + 1])) {
                        //left
                        //node->left = new bvh_node(true);
                        nodes[cur_idx] = bvh_node(true);
                        node->left_idx = cur_idx;
                        nodes[cur_idx].primitive = src_objects[current.start];
                        cur_idx++;

                        //node->left->primitive = src_objects[current.start];

                        //right
                        //node->right = new bvh_node(true);
                        nodes[cur_idx] = bvh_node(true);
                        node->right_idx = cur_idx;
                        nodes[cur_idx].primitive = src_objects[current.start + 1];
                        cur_idx++;
                        //node->right->primitive = src_objects[current.start + 1];
                    } else {
                        //left
                        //node->left = new bvh_node(true);
                        nodes[cur_idx] = bvh_node(true);
                        node->left_idx = cur_idx;
                        nodes[cur_idx].primitive = src_objects[current.start + 1];
                        cur_idx++;

                        //node->left->primitive = src_objects[current.start+1];

                        //right
                        //node->right = new bvh_node(true);
                        nodes[cur_idx] = bvh_node(true);
                        node->right_idx = cur_idx;
                        nodes[cur_idx].primitive = src_objects[current.start];
                        cur_idx++;

                        //node->right->primitive = src_objects[current.start];
                    }

                } else {

                    //sort hittables based on selected axis
                    quicksort_primitives(src_objects, int(current.start), int(current.end - 1), comparator);

                    //more than 2 nodes, create left and right nodes, connect them to current node and push them onto stack
                    //node->left = new bvh_node(false);
                    nodes[cur_idx] = bvh_node(false);
                    node->left_idx = cur_idx;
                    cur_idx++;

                    //node->right = new bvh_node(false);
                    nodes[cur_idx] = bvh_node(false);
                    node->right_idx = cur_idx;
                    cur_idx++;

                    auto mid = current.start + current_span / 2;

                    //push onto stack

                    //push left child
                    tos++;
                    if (tos >= MAX_DEPTH) {
                        delete[] node_stack;
                        delete[] stack;
                        return false;
                    }

                    stack[tos] = stack_item(current.start, mid);
                    //node_stack[tos] = node->left;
                    node_stack[tos] = &nodes[node->left_idx];

                    //push right child
                    tos++;
                    if (tos >= MAX_DEPTH) {
                        delete[] node_stack;
                        delete[] stack;
                        return false;
                    }

                    stack[tos] = stack_item(mid, current.end);
                    //node_stack[tos] = node->right;
                    node_stack[tos] = &nodes[node->right_idx];

                    n_nodes+=2;
                }
            }
        }

    }

    size = n_nodes;
    delete[] node_stack;
    delete[] stack;
    return true;
}

__device__ void bvh::build_nodes_bboxes() {
    
    int bbox_created = 0;
    bvh_node* current_node = &nodes[0];
    
    if(!current_node->is_leaf) { //at least two elements
        int tos = -1;
        auto node_stack = new bvh_node * [size];

        do {
            //traverse to leftmost inner node of subtree
            while (!current_node->is_leaf) {
                node_stack[++tos] = current_node;
                current_node = &nodes[current_node->left_idx];
            }

            //peek element on tos
            bvh_node* top_node = node_stack[tos];

            if (nodes[top_node->right_idx].is_leaf || nodes[top_node->right_idx].bbox.isValid()) {
                //compute bbox based on children
                bbox_created++;
                top_node->create_bbox(nodes);
                tos--; //actual pop
            }
            else {
                current_node = &nodes[top_node->right_idx];
            }

        } while (tos >= 0 || !current_node->is_leaf);

        delete[] node_stack;
    }

    
}
