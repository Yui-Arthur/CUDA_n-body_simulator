#version 330 core
layout (location = 0) in vec3 position;

uniform vec3 color;
uniform mat4 trans;
out vec3 c;
void main()
{
    gl_Position = trans * vec4(position.x, position.y, position.z, 1.0);
    c = color;
};