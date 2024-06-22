
#ifndef _N_BODY_ALGO_
#define _N_BODY_ALGO_

#include <stdlib.h>
#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "circle.h"
#include "n_body_cuda.h"
#include <vector>
#include <unistd.h>
#include <time.h>


#define CPU_COMPUTE 0
#define CUDA_COMPUTE 1
class n_body_algo{
public:
    circle circle_body;
    n_body_algo(char* fileName, GLFWwindow* window){
        srand(time(NULL));
        init_system(fileName, window);
    }
    void init_system(char* fileName, GLFWwindow* window){
        /*  initialize */
        FILE* fp = fopen(fileName, "r");
        
        fscanf(fp, "%d", &bodies_cnt);

        /* 質量masses 位置pos 速度vel 加速度acc */
        GLfloat masses; 
        glm::vec3 pos, vel, acc;
        pos.z = 0;

        printf("setting %d\n", bodies_cnt);

        v.resize(bodies_cnt, glm::vec3(0.0f));
        move_dis.resize(bodies_cnt, glm::vec3(0.0f));
        last_a.resize(bodies_cnt, glm::vec3(0.0f));

        circle_body = circle(100, "shader/circle.vert", "shader/circle.frag");
        // circle_body = circle();
        
        float r, g, b;
        
        
        for(int i = 0; i < bodies_cnt; i++){
            fscanf(fp, "%f", &masses);
            fscanf(fp, "%f%f", &pos.x, &pos.y);
            fscanf(fp, "%f%f", &vel.x, &vel.y);
            v[i].x = vel.x;
            v[i].y = vel.y;

            r = (rand() % 255) / 255.0f;
            usleep(100 * 1000);
            g = (rand() % 255) / 255.0f;
            usleep(100 * 1000);
            b = (rand() % 255) / 255.0f;
            circle_body.create_circle(pos.x, pos.y, v[i].x, v[i].y, masses, glm::vec3(r, g, b));

            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            circle_body.draw_circle();
            glfwSwapBuffers(window);
        }

        // Allocate memory on GPU
        cudaMalloc((void**)&d_center_pos, bodies_cnt * sizeof(glm::vec4));
        cudaMalloc((void**)&d_mass, bodies_cnt * sizeof(GLfloat));
        cudaMalloc((void**)&d_v, bodies_cnt * sizeof(glm::vec3));
        cudaMalloc((void**)&d_move_dis, bodies_cnt * sizeof(glm::vec4));
        cudaMalloc((void**)&d_slow_flag, sizeof(GLfloat));

        cudaMemcpy(d_center_pos, circle_body.center_pos.data(), bodies_cnt * sizeof(glm::vec4), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass, circle_body.mass.data(), bodies_cnt * sizeof(GLfloat), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, v.data(), bodies_cnt * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_GravConstant, &GravConstant, sizeof(GLfloat), cudaMemcpyHostToDevice);


        
        fclose(fp);
    }    

    ~n_body_algo(){
        cudaFree(d_center_pos);
        cudaFree(d_mass);
        cudaFree(d_v);
        cudaFree(d_slow_flag);
    }

    void next_timestamp(int iteration, bool compute_type){
        
        if( compute_type == CUDA_COMPUTE){
            if(iteration != 0){
                cuda_compute(iteration);
                circle_body.move(v);
            }
        }
        else{
            for(int i=0; i<iteration; i++){
                compute();
                circle_body.move(v);
            }
        }

        
        circle_body.draw_circle();
        // sleep(1);
    }

    void show_infomation(){
        for(int i=0; i<bodies_cnt; i++){
            printf("%d:%f  pos (%f, %f), v (%f, %f), a(%f, %f)\n", 
                i, circle_body.mass[i], circle_body.center_pos[i].x, circle_body.center_pos[i].y,
                v[i].x, v[i].y, last_a[i].x, last_a[i].y);
        }
    }
    
private:
    const GLfloat GravConstant = 6.67430 * 1e-5;
    int bodies_cnt;
    std::vector<glm::vec3> v;
    std::vector<glm::vec3> move_dis;
    std::vector<glm::vec3> last_a;

    glm::vec4* d_center_pos;
    float* d_mass;
    glm::vec3* d_v;
    glm::vec3 *d_move_dis;
    int *d_slow_flag;


    void compute(){
        std::vector<glm::vec3> a(bodies_cnt, glm::vec3(0.0f));
        bool slow_down_flag = false;
        for(int i = 0; i < bodies_cnt; i++){
            // glm::vec3 a(0.0f);
            
            for(int j = 0; j < bodies_cnt; j++){
                
                glm::vec4 dis = circle_body.center_pos[j] - circle_body.center_pos[i];
                GLfloat denominator = glm::dot(dis, dis);
                denominator += 0.000001f;
                // glm::vec4 test = body.center_pos[i] - body.center_pos[j];
                // printf("i %f, %f, %f, %f\n", body.center_pos[i].w, body.center_pos[i].x, body.center_pos[i].y, body.center_pos[i].z);
                // printf("j %f, %f, %f, %f\n", body.center_pos[j].w, body.center_pos[j].x, body.center_pos[j].y, body.center_pos[j].z);
                // printf("diff %f, %f, %f, %f\n", test.w, test.x, test.y, test.z);
                // printf("dot %f\n", denominator);
                denominator = glm::pow(denominator, 3/2.0f);
                // printf("deno %f\n", denominator);

                glm::vec4 numerator = dis;
                numerator = numerator * circle_body.mass[j];
                // printf("num %f, %f, %f, %f\n", numerator.w, numerator.x, numerator.y, numerator.z);

                glm::vec4 result = (GravConstant * numerator) / denominator;
                // printf("result %f, %f, %f, %f\n", result.w, result.x, result.y, result.z);
                a[i] += glm::vec3(result);
                // printf("\n");

                if(i != j && glm::pow(glm::dot(dis, dis), 2) < 100.0f)
                    slow_down_flag = true;
            }
            // printf("final acc %f, %f, %f\n\n", a.x, a.y, a.z);
        }

        if(slow_down_flag)
            return;

        for(int i = 0; i < bodies_cnt; i++){
            v[i] += a[i];
            last_a[i] = a[i];
        }

        
    }


    void cuda_compute(int iteration){
        int threadsPerBlock = 256;
        int blocksPerGrid = (bodies_cnt + threadsPerBlock - 1) / threadsPerBlock;
        int shared_memory_size = sizeof(glm::vec4) * bodies_cnt;

        // printf("<<<%d, %d>>> shared %d\n", blocksPerGrid, threadsPerBlock, shared_memory_size);
        
        cudaError R;
        compute_CUDA<<<blocksPerGrid, threadsPerBlock, shared_memory_size>>>(d_center_pos, d_mass, d_v, d_move_dis, bodies_cnt, GravConstant, d_slow_flag, iteration);

        // R = cudaGetLastError();
        // printf("kernel func start / Cuda Error : %s\n",cudaGetErrorString(R));
        // R = cudaDeviceSynchronize();
        // printf("kernel func run / Cuda Error : %s\n",cudaGetErrorString(R));
        cudaDeviceSynchronize();


        cudaMemcpy(move_dis.data(), d_move_dis, bodies_cnt * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        cudaMemcpy(v.data(), d_v, bodies_cnt * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

        circle_body.move(move_dis);

    }
};


#endif