#include <stdio.h>
#include <stdlib.h>
#include "bmp_header.h"
#include <string.h>
#include "math.h"
#include <time.h>
#include "mpi.h"
#include <omp.h>

#define MASTER 0

using namespace std;

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

void black_white(pixel **bitmap, int height, int width, int rank, int numprocs, MPI_Datatype MPI_PIXEL) {
    int start = rank * ((double) height) / numprocs;
    int end = min((int)((rank + 1) * ((double) height) / numprocs), height);
    int i, j, val;
    #pragma omp parallel for default(shared) private(i, j, val)
    for (i = start; i < end; i++) {
        for (j = 0; j < width; j++) {
            val = (bitmap[i][j].r + bitmap[i][j].g + bitmap[i][j].b) / 3;
            bitmap[i][j].r = val;
            bitmap[i][j].g = val;
            bitmap[i][j].b = val;
        }
    }
    if (rank == MASTER) {
        for (i = 1; i < numprocs; i++) {
            int procstart = i * ((double) height) / numprocs;
            int procend = min((int)((i + 1) * ((double) height) / numprocs), height);
            for (j = procstart; j < procend; j++) {
                MPI_Recv(&(bitmap[j][0]), width, MPI_PIXEL, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (i = 1; i < numprocs; i++) {
            for (j = 0; j < height; j++)
                MPI_Send(&(bitmap[j][0]), width, MPI_PIXEL, i, 0, MPI_COMM_WORLD);
        }
        
    } else {
        for (i = start; i < end; i++) {
            MPI_Send(&(bitmap[i][0]), width, MPI_PIXEL, MASTER, 0, MPI_COMM_WORLD);
        }
        for (j = 0; j < height; j++)
            MPI_Recv(&(bitmap[j][0]), width, MPI_PIXEL, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

pixel **sobel_edge_detection(pixel **bitmap, int height, int width, int rank, int numprocs, MPI_Datatype MPI_PIXEL) {
    int xkernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int ykernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    int i, j, x, y;
    pixel **new_map = (pixel **)malloc(height * sizeof(pixel *));
    for (i = 0; i < height; i++)
        new_map[i] = (pixel *)malloc(width * sizeof(pixel));

    black_white(bitmap, height, width, rank, numprocs, MPI_PIXEL);

    int start = rank * ((double) height) / numprocs;
    int end = min((int)((rank + 1) * ((double) height) / numprocs), height);
    #pragma omp parallel for default(shared) private(i, j, x, y)
    for (i = start; i < end; i++) {
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
    if (rank == MASTER) {
        for (i = 1; i < numprocs; i++) {
            int procstart = i * ((double) height) / numprocs;
            int procend = min((int)((i + 1) * ((double) height) / numprocs), height);
            for (int j = procstart; j < procend; j++) {
                MPI_Recv(&(new_map[j][0]), width, MPI_PIXEL, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (i = 1; i < numprocs; i++) {
            for (int j = 0; j < height; j++)
                MPI_Send(&(new_map[j][0]), width, MPI_PIXEL, i, 0, MPI_COMM_WORLD);
        }
        
    } else {
        for (i = start; i < end; i++) {
            MPI_Send(&(new_map[i][0]), width, MPI_PIXEL, MASTER, 0, MPI_COMM_WORLD);
        }
        for (j = 0; j < height; j++)
            MPI_Recv(&(new_map[j][0]), width, MPI_PIXEL, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return new_map;
}

MPI_Datatype create_MPI_pixel_type() {
    const int nfields = 3;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR};
    MPI_Datatype mpi_pixel_type;
    MPI_Aint     offsets[3];
    offsets[0] = offsetof(pixel, r);
    offsets[1] = offsetof(pixel, g);
    offsets[2] = offsetof(pixel, b);

    MPI_Type_create_struct(nfields, blocklengths, offsets, types, &mpi_pixel_type);
    MPI_Type_commit(&mpi_pixel_type);
    return mpi_pixel_type;
}

int main(int argc, char *argv[]) {

    int rank, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if (argc != 2) {
        if (rank == MASTER)
            printf("Usage: mpirun -np <num_procs> mpi-sobel <bmp_file_name>\n");
        MPI_Finalize();
        return 0;
    }
    int i, j, width, height;
    FILE *input;
    bmp_fileheader fh;
    bmp_infoheader ih;
    pixel **bitmap;
    clock_t t;

    MPI_Datatype MPI_PIXEL = create_MPI_pixel_type();
    if (rank == MASTER) {
        input = fopen(argv[1], "rb");
        bitmap = read_bmp(&fh, &ih, input);
        width = ih.width;
        height = ih.height;

        // send the bitmap to all procs
        for (i = 1; i < numprocs; i++) {
            MPI_Send(&height, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&width, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            for (j = 0; j < height; j++)
                MPI_Send(&(bitmap[j][0]), width, MPI_PIXEL, i, 0, MPI_COMM_WORLD);
        }

        t = clock();
    } else {
        MPI_Recv(&height, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&width, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        bitmap = (pixel **)malloc(height * sizeof(pixel *));
        for (i = 0; i < height; i++)
            bitmap[i] = (pixel *)malloc(width * sizeof(pixel));

        for (j = 0; j < height; j++)
            MPI_Recv(&(bitmap[j][0]), width, MPI_PIXEL, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    pixel **new_map = sobel_edge_detection(bitmap, height, width, rank, numprocs, MPI_PIXEL);

    if (rank == MASTER) {
        t = clock() - t;
        FILE *output = fopen("sobel_hybrid1.bmp", "wb");
        write_bmp(fh, ih, new_map, output);

        double duration = ((double)t)/CLOCKS_PER_SEC;
        printf("Time: %f\n", duration);

        fclose(output);
        fclose(input);
    }
    for (int i = 0; i < height; i++) {
        free(bitmap[i]);
        free(new_map[i]);
    }
    free(bitmap);
    free(new_map);

    MPI_Finalize();
    return 0;
}
