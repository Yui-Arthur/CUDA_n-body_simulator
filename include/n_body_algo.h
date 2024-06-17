
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
#include <vector>
#include <unistd.h>
#include <time.h>

class n_body_algo{
public:
    n_body_algo(char* fileName){
        init_system(fileName);
        srand(time(NULL));
    }
    void init_system(char* fileName){
        /*  initialize */
        FILE* fp = fopen(fileName, "r");
        
        fscanf(fp, "%f%d", &GravConstant, &bodies);

        /* 質量masses 位置pos 速度vel 加速度acc */
        GLfloat masses; 
        glm::vec3 pos, vel, acc;

        printf("setting %f, %d\n", GravConstant, bodies);

        v.resize(bodies, glm::vec3(0.0f));

        body = circle(100, "shader/circle.vert", "shader/circle.frag");
        
        float r, g, b;
        
        
        for(int i = 0; i < bodies; i++){
            fscanf(fp, "%f", &masses);
            fscanf(fp, "%f%f%f", &pos.x, &pos.y, &pos.z);
            // fscanf(fp, "%lf%lf%lf", &vel.x, &vel.y, &vel.z);
            r = (rand() % 255) / 255.0f;
            usleep(100 * 1000);
            r = (rand() % 255) / 255.0f;
            usleep(100 * 1000);
            b = (rand() % 255) / 255.0f;
            body.create_circle(pos.x, pos.y, masses, glm::vec3(r, g, b));
        }

        
        fclose(fp);
    }    

    void next_timestamp(){
        compute();
        body.move(v);
        body.draw_circle();
        // sleep(1);
    }

    void compute(){
        
        for(int i = 0; i < bodies; i++){
            glm::vec3 a(0.0f);
            
            for(int j = 0; j < bodies; j++){
                
                
                GLfloat denominator = glm::dot(body.center_pos[j] - body.center_pos[i], body.center_pos[j] - body.center_pos[i]);
                denominator += 0.000001f;
                glm::vec4 test = body.center_pos[i] - body.center_pos[j];
                // printf("i %f, %f, %f, %f\n", body.center_pos[i].w, body.center_pos[i].x, body.center_pos[i].y, body.center_pos[i].z);
                // printf("j %f, %f, %f, %f\n", body.center_pos[j].w, body.center_pos[j].x, body.center_pos[j].y, body.center_pos[j].z);
                // printf("diff %f, %f, %f, %f\n", test.w, test.x, test.y, test.z);
                // printf("dot %f\n", denominator);
                denominator = glm::pow(denominator, 3/2.0f);
                // printf("deno %f\n", denominator);

                glm::vec4 numerator = body.center_pos[j] - body.center_pos[i];
                numerator = numerator * body.mass[j];
                // printf("num %f, %f, %f, %f\n", numerator.w, numerator.x, numerator.y, numerator.z);

                glm::vec4 result = (1e3f * GravConstant * numerator) / denominator;
                // printf("result %f, %f, %f, %f\n", result.w, result.x, result.y, result.z);
                a += glm::vec3(result);
                // printf("\n");
            }
            printf("final acc %f, %f, %f\n\n", a.x, a.y, a.z);
            v[i] += a;
        }
    }

private:
    GLfloat GravConstant;
    int bodies;
    circle body;
    std::vector<glm::vec3> v;
};


#endif