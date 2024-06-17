#ifndef _CIRCLE_H_
#define _CIRCLE_H_

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include "shader_class.h"

class circle{
    public:
        std::vector<GLfloat*> circle_pointer;
        std::vector<glm::vec4> center_pos;
        std::vector<glm::vec4> show_pos;
        std::vector<GLfloat> mass;
        std::vector<glm::vec3> color;
        std::vector<GLuint*> VAO;
        std::vector<GLuint*> VBO;
        std::vector<glm::mat4> translation;
        Shader circal_shader;
        
        int num_trians, num_segments;

        circle(){};
        circle(int num_s, const GLchar* vertexPath, const GLchar* fragmentPath){
            circal_shader = Shader(vertexPath, fragmentPath);
            num_segments = num_s;
            num_trians = num_segments - 2;
            printf("circle init success\n");
        }
        void clear_VAO_VBO(){
            for(int i=0; i<VAO.size(); i++){
                glDeleteVertexArrays(1, VAO[i]);
                glDeleteBuffers(1, VBO[i]);
            }
        }

        void move(std::vector<glm::vec3> velocities){
            for(int i=0; i<circle_pointer.size(); i++){
                
                translation[i] = glm::translate(translation[i], (velocities[i] / 800.0f));
                center_pos[i] =  center_pos[i] + glm::vec4(velocities[i], 0.0f);
            }

            // printf("\n");
            return;
        }

        void create_circle(float cx, float cy, float r, glm::vec3 c){
            build_circle(cx, cy, r);
            bind_circle();
            color.push_back(c);
            mass.push_back(800*r);
            center_pos.push_back(glm::vec4(800*cx, 800*cy, 0.0f, 1.0f));
            show_pos.push_back(glm::vec4(cx, cy, 0.0f, 1.0f));
            translation.push_back(glm::mat4(GLfloat(1.0f)));
        
            printf("create_circle pos (%f, %f), mass %f\n", cx, cy, r);

            // for(int i=0; i<num_trians*3; i++){
            //     printf("%f, %f, %f\n", circle_pointer[circle_pointer.size()-1][i], circle_pointer[circle_pointer.size()-1][i+1], circle_pointer[circle_pointer.size()-1][i+2]);
            // }
        }
        void draw_circle(){
            glm::vec3 c;
            glm::mat4 t;
            GLint c_loc = glGetUniformLocation(circal_shader.Program, "color"); 
            GLint t_loc = glGetUniformLocation(circal_shader.Program, "trans"); 
            circal_shader.Use();

            // printf("%d\n", circle_pointer.size());
            // printf("%d\n", color.size());
            // printf("%d\n", VAO.size());
            // printf("%d\n", VBO.size());

            
            for(int i=0; i<VAO.size(); i++){
                c = color[i];
                t = translation[i];
                // printf("%f, %f, %f\n",t[0][0], t[1][1], t[2][2]);
                glUniform3fv(c_loc, 1, glm::value_ptr(c));
                glUniformMatrix4fv(t_loc, 1, GL_FALSE, glm::value_ptr(t));
                glBindVertexArray(*(VAO[i]));
                // printf("%d\n", i);
                glDrawArrays(GL_TRIANGLES, 0, num_trians*3);
                glBindVertexArray(0);
                // break;

                glm::vec4 tmp(show_pos[i]);
                tmp = translation[i] * tmp ;
                printf("really pos %f, %f, %f\n", center_pos[i].x, center_pos[i].y, center_pos[i].z);
                printf("normalize pos %f, %f, %f\n", center_pos[i].x / 800.f, center_pos[i].y / 800.f, center_pos[i].z / 800.f);
                printf("show pos %f, %f, %f\n", tmp.x , tmp.y, center_pos[i].z);
            }
            
        }
    private:
        void build_circle(float cx, float cy, float r)
        {
            float theta = 3.1415926 * 2 / float(num_segments);
            float tangetial_factor = tanf(theta);//calculate the tangential factor 

            float radial_factor = cosf(theta);//calculate the radial factor 

            float x = r;//we start at angle = 0 

            float y = 0;
            GLfloat *vertices = (GLfloat*)malloc(num_trians * 9 * sizeof(GLfloat));

            GLfloat tmp[num_segments*3];
            circle_pointer.push_back(vertices);
            
            for (int ii = 0; ii < num_segments; ii++)
            {
                tmp[3*ii] = x + cx;
                tmp[3*ii+1] = y + cy;
                tmp[3*ii+2] = 0;

                float tx = -y;
                float ty = x;

                x += tx * tangetial_factor;
                y += ty * tangetial_factor;

                x *= radial_factor;
                y *= radial_factor;
            }

            for(int ii = 0; ii < num_trians; ii++){
                vertices[9*ii+0] = tmp[0];
                vertices[9*ii+1] = tmp[1];
                vertices[9*ii+2] = tmp[2];

                vertices[9*ii+3] = tmp[3*ii+3];
                vertices[9*ii+4] = tmp[3*ii+4];
                vertices[9*ii+5] = tmp[3*ii+5];

                vertices[9*ii+6] = tmp[3*ii+6];
                vertices[9*ii+7] = tmp[3*ii+7];
                vertices[9*ii+8] = tmp[3*ii+8];

            }

            return;
        }

        void bind_circle(){
            GLuint *tmp_VAO = new GLuint;
            GLuint *tmp_VBO = new GLuint;
            glGenVertexArrays(1, tmp_VAO);
            glGenBuffers(1, tmp_VBO);

            glBindVertexArray(*tmp_VAO);
            glBindBuffer(GL_ARRAY_BUFFER, *tmp_VBO);
            glBufferData(GL_ARRAY_BUFFER, num_trians * 9 * sizeof(GLfloat), circle_pointer[circle_pointer.size()-1], GL_DYNAMIC_DRAW);
            // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_TRUE, 3 * sizeof(GLfloat), (GLvoid*)0);
            glEnableVertexAttribArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0); 
            glBindVertexArray(0);

            VAO.push_back(tmp_VAO);
            VBO.push_back(tmp_VBO);
        }


};

#endif
