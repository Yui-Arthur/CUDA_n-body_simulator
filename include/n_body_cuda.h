#ifndef _N_BODY_CUDA_
#define _N_BODY_CUDA_

#include <stdlib.h>
#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <unistd.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>    
 
__global__ void compute_CUDA(glm::vec4* center_pos, float* mass, glm::vec3* v, glm::vec3* move_dis, int bodies, GLfloat GravConstant, int *slow_flag, int  iteration){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= bodies){
        return;
    }

    extern __shared__ glm::vec4 shared_circle_pos[];
    
    glm::vec4 origin_pos = shared_circle_pos[i];

    shared_circle_pos[i] = center_pos[i];
    (*slow_flag) = false;
    __syncthreads();

    for(int _=0; _<iteration; _++){

        glm::vec3 a(0.0f);
        for(int j = 0; j < bodies; j++){

            glm::vec3 dis = shared_circle_pos[j] - shared_circle_pos[i];

            GLfloat denominator = glm::dot(dis, dis);
            // GLfloat denominator = glm::dot(glm::vec3(shared_circle_pos[j] - shared_circle_pos[i]), glm::vec3(shared_circle_pos[j] - shared_circle_pos[i]));
            denominator += 0.000001f;
            denominator = glm::pow(denominator, 3/2.0f);
            glm::vec4 test = shared_circle_pos[i] - shared_circle_pos[j];
            // printf("diff %f, %f, %f, %f\n", test.w, test.x, test.y, test.z);
            // printf("i %f, %f, %f, %f\n", center_pos[i].w, center_pos[i].x, center_pos[i].y, center_pos[i].z);
            // printf("j %f, %f, %f, %f\n", center_pos[j].w, center_pos[j].x, center_pos[j].y, center_pos[j].z);
            // printf("diff %f, %f, %f, %f\n", test.w, test.x, test.y, test.z);
            // printf("dot %f\n", denominator);
            glm::vec4 numerator = shared_circle_pos[j] - shared_circle_pos[i];
            numerator = numerator * mass[j];
            glm::vec4 result =  (GravConstant * numerator) / denominator;
            a += glm::vec3(result);

            if(i != j && glm::pow(glm::dot(dis, dis), 2) < 100.0f)
                (*slow_flag) = true;
        } 

        
        // printf("final acc %f, %f, %f\n\n", a.x, a.y, a.z);

        __syncthreads();
        
        if((*slow_flag) == false){
            v[i] += a;
        }
        shared_circle_pos[i] = shared_circle_pos[i] + glm::vec4(v[i], 0.0f);
        (*slow_flag) = false;
        __syncthreads();

    }
    /* can't use for over one block */
    move_dis[i].x = (shared_circle_pos[i] - center_pos[i]).x;
    move_dis[i].y = (shared_circle_pos[i] - center_pos[i]).y;
    move_dis[i].z = 0.0f;

    center_pos[i] = shared_circle_pos[i];

    
    

}

#endif