#version 460 core

out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;

// texture sampler || 纹理单元(Texture Unit)
// uniform sampler2D texture1; /* first */
// uniform sampler2D ourTexture; /* second */

/* 两个纹理的结合 */
uniform sampler2D texture1;
uniform sampler2D texture2;

void main()
{
	 /* first */
	// FragColor = texture(texture1, TexCoord);

	 /* second */
	// FragColor = texture(ourTexture, TexCoord) * vec4(ourColor, 1.0);

	/* 两个纹理的结合 */
	FragColor = mix(texture(texture1, TexCoord), texture(texture2, TexCoord), 0.2);
}
