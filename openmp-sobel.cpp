#include <stdio.h>
#include <stdlib.h>
#include "bmp_header.h"
#include "math.h"
#include <omp.h>
#include <time.h>

pixel **read_bmp(bmp_fileheader *fh, bmp_infoheader *ih, FILE *input)
{
    int i, j;

    fread(&fh->fileMarker1, sizeof(unsigned char), 1, input);
    fread(&fh->fileMarker2, sizeof(unsigned char), 1, input);
    fread(&fh->bfSize, sizeof(unsigned int), 1, input);
    fread(&fh->unused1, sizeof(unsigned short), 1, input);
    fread(&fh->unused2, sizeof(unsigned short), 1, input);
    fread(&fh->imageDataOffset, sizeof(unsigned int), 1, input);

    fread(&ih->biSize, sizeof(unsigned int), 1, input);
    fread(&ih->width, sizeof(signed int), 1, input);
    fread(&ih->height, sizeof(signed int), 1, input);
    fread(&ih->planes, sizeof(unsigned short), 1, input);
    fread(&ih->bitPix, sizeof(unsigned short), 1, input);
    fread(&ih->biCompression, sizeof(unsigned int), 1, input);
    fread(&ih->biSizeImage, sizeof(unsigned int), 1, input);
    fread(&ih->biXPelsPerMeter, sizeof(int), 1, input);
    fread(&ih->biYPelsPerMeter, sizeof(int), 1, input);
    fread(&ih->biClrUsed, sizeof(unsigned int), 1, input);
    fread(&ih->biClrImportant, sizeof(unsigned int), 1, input);

    pixel **bitmap = (pixel **)malloc(ih->height * sizeof(pixel *));
    for (i = 0; i < ih->height; i++)
        bitmap[i] = (pixel *)malloc(ih->width * sizeof(pixel));
    if (bitmap == NULL)
        exit(-1);

    fseek(input, fh->imageDataOffset, SEEK_SET);
    for (i = 0; i < ih->height; i++)
    {
        for (j = 0; j < ih->width; j++)
        {
            fread(&bitmap[i][j].b, sizeof(char), 1, input);
            fread(&bitmap[i][j].g, sizeof(char), 1, input);
            fread(&bitmap[i][j].r, sizeof(char), 1, input);
        }
        if ((ih->width * 3) % 4 != 0)
            fseek(input, 4 - ((ih->width * 3) % 4), SEEK_CUR);
    }
    return bitmap;
}

void write_bmp(bmp_fileheader fh, bmp_infoheader ih, pixel **bitmap, FILE *output)
{
    int i, j;
    fwrite(&fh, sizeof(bmp_fileheader), 1, output);
    fwrite(&ih, sizeof(bmp_infoheader), 1, output);
    fwrite("\0", sizeof(char), fh.imageDataOffset - ftell(output), output);
    for (i = 0; i < ih.height; i++)
    {
        for (j = 0; j < ih.width; j++)
        {
            fwrite(&bitmap[i][j].b, sizeof(char), 1, output);
            fwrite(&bitmap[i][j].g, sizeof(char), 1, output);
            fwrite(&bitmap[i][j].r, sizeof(char), 1, output);
        }
        if ((ih.width * 3) % 4 != 0)
            for (j = 0; j < 4 - ((ih.width * 3) % 4); j++)
                fwrite("\0", sizeof(char), 1, output);
    }
}

void black_white(pixel **bitmap, int height, int width) {
    int i, j, val;
    
    #pragma omp parallel for default(shared) private(i, j, val)
    for (i = 0; i < height; i++)
        for (j = 0; j < width; j++) {
            val = (bitmap[i][j].r + bitmap[i][j].g + bitmap[i][j].b) / 3;
            bitmap[i][j].r = val;
            bitmap[i][j].g = val;
            bitmap[i][j].b = val;
        }
}

pixel **sobel_edge_detection(pixel **bitmap, int height, int width) {
    int xkernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int ykernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    int i, j, x, y;
    pixel **new_map = (pixel **)malloc(height * sizeof(pixel *));
    for (i = 0; i < height; i++)
        new_map[i] = (pixel *)malloc(width * sizeof(pixel));

    black_white(bitmap, height, width);

    #pragma omp parallel for default(shared) private(i, j, x, y)
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            int xval = 0;
            int yval = 0;

            if ((i > 0) && (j > 0) && (i < height - 1) && (j < width - 1)) {

                for (x = -1; x <= 1; x++) {
                    for (y = -1; y <= 1; y++) {
                        xval += bitmap[i + x][j + y].r * xkernel[x + 1][y + 1];
                        yval += bitmap[i + x][j + y].r * ykernel[x + 1][y + 1];
                    }
                }
            }
            int val = sqrt(xval * xval + yval * yval);
            if (val > 255)
                val = 255;
            if (val < 0)
                val = 0;
            new_map[i][j].r = val;
            new_map[i][j].g = val;
            new_map[i][j].b = val;
        }
    }
    return new_map;
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Usage: ./openmp-sobel <bmp_file_name>\n");
        return 0;
    }
    FILE *input = fopen(argv[1], "rb");
    bmp_fileheader fh;
    bmp_infoheader ih;
    pixel **bitmap = read_bmp(&fh, &ih, input);

    clock_t t = clock();
    pixel **new_map = sobel_edge_detection(bitmap, ih.height, ih.width);
    t = clock() - t;

    FILE *output = fopen("sobel_openmp.bmp", "wb");
    write_bmp(fh, ih, new_map, output);

    double duration = ((double)t)/CLOCKS_PER_SEC;
    printf("Time: %f\n", duration);

    for (int i = 0; i < ih.height; i++) {
        free(bitmap[i]);
        free(new_map[i]);
    }
    free(bitmap);
    free(new_map);
    fclose(output);
    fclose(input);
    return 0;
}
