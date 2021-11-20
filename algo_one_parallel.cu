#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
// #define size 1024

#define CUDA_WARN(XXX) \
    do {if (XXX != cudaSuccess) printf("%s\n", cudaGetErrorString(XXX));} while (0)

#define BLOCK_ROW 32
#define BLOCK_COL 32

__global__ void kernal_process_image(float* image, int height, int width, int kernel)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row>0 && col>0 && row<height-1 && col<width-1 && image[row*width+col]==0)
    {
        float div=0, su=0, wei=0;
        int i, j;
        for(i=-1; i<=1; i++)
            for(j=-1; j<=1; j++)
                div+=image[(row+i)*width+col+j];

        if(div>0.06)
        {
            int range = kernel/2;
            for(i=-range; i<=range; i++){
                for(j=-range;j<=range;j++){
                    if(row+i<0 || row+i>height || col+j<0 || col+j>width || image[(row+i)*width+col+j]==0){
                        continue;
                    }
                    wei += 1 / sqrt((i*i) + (j*j));
                    su += (image[(row+i)*width+col+j]/sqrt((i*i) + (j*j)));
                }
            }
        }
        image[row*width+col]=(wei!=0)?su/wei:0;
        // printf("%d,", row*width+col);
    }
    return;
}

void process_image(float* input_image, int height, int width)
{
    float* image;

    cudaMalloc(&image, height*width*sizeof(float));

    cudaMemcpy(image, input_image, height*width*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_ROW, BLOCK_COL);
    dim3 dimGrid((height-1)/dimBlock.x + 1, (width-1)/dimBlock.y + 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    kernal_process_image <<<dimBlock, dimGrid>>> (image, height, width, 5);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time=0;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("Time to identify black pixels and replace those with weighted average for image size : %d x %d is: %f miliseconds\n",height,width,elapsed_time);

    cudaMemcpy(input_image, image, height*width*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(image);
    return;
}


int main(int argc, char **argv)
{
    int i, j, x;
	clock_t start,end;
	unsigned char byte[54];

    if(argc<3)
    {
        printf("Insufficient input Argument");
        return 1;
    }

	start = clock();

	FILE* fIn = fopen(argv[1], "rb");//Input File name
	FILE* fOut = fopen(argv[2], "wb");//Output File name


	if (fIn == NULL)											// check if the input file has not been opened succesfully.
	{
		printf("File does not exist.\n");
	}

	end = clock();
	double walltime=(double)(((double)(end-start)* 1000)/(double)CLOCKS_PER_SEC);
    printf("Time to open input & output image is: %f miliseconds\n",walltime);


    start = clock();
	for (i = 0; i < 54; i++)											//read the 54 byte header from fIn
	{
		byte[i] = getc(fIn);
	}

    unsigned int width = *(int*)&byte[18];
	unsigned int height = *(int*)&byte[22];
	fwrite(byte, sizeof(unsigned char), 54, fOut);					//write the header back

	end = clock();
	walltime=(double)(((double)(end-start)* 1000)/(double)CLOCKS_PER_SEC);
    printf("Time to read & write header file for size : %d x %d is: %f miliseconds\n",height,width,walltime);

	printf("width: %d\n", width);
	printf("height: %d\n", height);

    int size = height*width;

	unsigned char* buffer = (unsigned char*)malloc(size * sizeof(unsigned char));
	unsigned char* out = (unsigned char*)malloc(size * sizeof(unsigned char));
	float* c = (float*)malloc(size * sizeof(float));

	start = clock();
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			buffer[i * width + j] = getc(fIn);
		}
	}

    end = clock();
	walltime=(double)(((double)(end-start)* 1000)/(double)CLOCKS_PER_SEC);
    printf("Time to read image file into buffer for size : %d x %d is: %f miliseconds\n",height,width,walltime);


    start = clock();
	for (i = 0; i < size; i++)
	{
		c[i] = ((float)(buffer[i])) / (255.0f);
	}
	end = clock();
	walltime=(double)(((double)(end-start)* 1000)/(double)CLOCKS_PER_SEC);
    printf("Time to convert pixel values in 0-1 range for image size : %d x %d is: %f miliseconds\n",height,width,walltime);

    // Process image
    process_image(c, height, width);

	start = clock();
	for (i = 0; i < size; i++)
	{
		x = (int)(c[i]*255.0f);
		out[i] = (unsigned char)x;
	}

	fwrite(out, sizeof(unsigned char), size, fOut);           //write image data back to the file
    end = clock();
    walltime=(double)(((double)(end-start)* 1000)/(double)CLOCKS_PER_SEC);
    printf("Time to convert pixel values in range of 0-255 & write image in output for image size : %d x %d is: %f miliseconds\n",height,width,walltime);

	fclose(fIn);
	fclose(fOut);

    free(c);
    free(buffer);
    free(out);
	return 0;
}
