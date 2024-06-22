#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "include/n_body_algo.h"
#include <time.h>
 
// Function prototypes
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);

// Window dimensions
const GLuint WIDTH = 1000, HEIGHT = 1000;


// The MAIN function, from here we start the application and run the game loop
GLFWwindow* init(){
     // Init GLFW
    glfwInit();
    // Set all the required options for GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    
    // Create a GLFWwindow object that we can use for GLFW's functions
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "N-body-siumlator", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return NULL;
    }
    // Set the required callback functions
    glfwSetKeyCallback(window, key_callback);

    
    // Define the viewport dimensions
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);  
    glViewport(0, 0, width, height);

    return window;
}

int iteration = 100;
int total_iteration = 0;
bool print_info = false;
int main(int argc, char **argv)
{
    // char *p;
    char *filename = argv[1];
    const char *compute_type = argv[2];
    // int Xh = strtol(argv[2], &p, 10);
    GLFWwindow* window = init();
    n_body_algo n(filename, window);
    
   

    
    
    // Game loop
    while (!glfwWindowShouldClose(window))
    {
        // Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
        glfwPollEvents();

        // Render
        // Clear the colorbuffer
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        // glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // c.draw_circle();

        clock_t cal_start = clock();

        if(compute_type[0] == '1')
            n.next_timestamp(iteration, CUDA_COMPUTE);
        else
            n.next_timestamp(iteration, CPU_COMPUTE);

        clock_t cal_end = clock();
        float cal_elapsedTime = ((cal_end-cal_start)/(double)(CLOCKS_PER_SEC))/(double)(iteration);
        
        
        
        total_iteration += iteration;

        if(print_info){
            n.show_infomation();
            print_info = false;

            printf("total iteration : %d, jump iteration %d\n", total_iteration, iteration);
            if(compute_type[0] == '1')
                printf("Calculation Time on CUDA: %10.10f ms\n", cal_elapsedTime * 1000) ;
            else
                printf("Calculation Time on CPU: %10.10f ms\n", cal_elapsedTime * 1000) ;
            
            printf("FPS = %.1f\n", 1000 / (cal_elapsedTime * 1000 * iteration));
            // printf("Calculation Time on : %10.10f ms\n", cal_elapsedTime * 1000);
            printf("\n");
        }

        glfwSwapBuffers(window);
        // c.move(v);
    }

    // c.clear_VAO_VBO();
    glfwTerminate();
    return 0;
}

// Is called whenever a key is pressed/released via GLFW
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
    else if(key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
        iteration += 100;
    else if(key == GLFW_KEY_LEFT && action == GLFW_PRESS)
        iteration -= 100, iteration = max(0, iteration);
    else if(key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        print_info = true;

    

}