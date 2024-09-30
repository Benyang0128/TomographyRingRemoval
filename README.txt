1. Each parallel algorithms (OMP, OpenCL, CUDA) will has own README.txt to explain how to execute/use the code

2. For *Full Resources* will in Google Drive: https://drive.google.com/drive/folders/17wsbPj0KJdxChuP94zo_JelPyoT69hOb?usp=drive_link

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

OMP Information:

## The executable is included together with the source code
## Executable named 'recon_ring_remover_v6'

## If prefer to compile yourself, run the following command (Windows might need different option to compile with OpenCV):
g++ -w -o recon_ring_remover_v6 main.cpp image_filters.cpp image_transforms.cpp image_io.cpp -fopenmp `pkg-config --cflags --libs opencv4`

## Run the executable with following command:
# Unix-based system:
./recon_ring_remover_v6 input/ output/ rec_ out_openmp_ 1 1 0 0 30 -300 300 300 30 1

# Windows system:
recon_ring_remover_v6 input/ output/ rec_ out_openmp_ 1 1 0 0 30 -300 300 300 30 1

# Hint usage
recon_ring_remover [input path] [output path] [input root] [output root] [first file num] [last file num] [center x y] [max ring width] [thresh min max] [ring threshold] [angular min] [verbose]

## Run the verification code
python image_compare.py

## Run performance output and plotting:
python run_computation.py

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

OpenCL Information:

Run OpenCL, locate file  (the file location):
cd C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL

-----------------------------------------------------------------------------------------------------

Setup G++, need to locate your computer tiff library and OpenCL location:
g++ -o recon_ring_remover.1.1.0 main.cpp image_filters.cpp image_transforms.cpp image_io.cpp -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64" -I"C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\vcpkg\installed\x64-windows\include" -L"C:\Users\tbeny\OneDrive\Desktop\RingRemoval_OpenCL\vcpkg\installed\x64-windows\lib" -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -ltiff -lOpenCL -Wno-deprecated-declarations

-----------------------------------------------------------------------------------------------------

Test Out:
recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 1 1 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 2 2 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 3 3 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 4 4 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 5 5 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 6 6 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 7 7 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 8 8 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 9 9 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 10 10 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 11 11 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 12 12 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 13 13 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 14 14 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 15 15 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 16 16 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 20 20 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 21 21 0 0 30 -300 300 300 30 1

recon_ring_remover.1.1.0 ./InputImages/ ./OutputImages/ rec_ out_opencl_ 22 22 0 0 30 -300 300 300 30 1

-----------------------------------------------------------------------------------------------------

python run verification (image compare):
python image_compare.py

-----------------------------------------------------------------------------------------------------

python run matplot:
python run_computation.py

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CUDA Information:

1.Run CudaRuntime4, locate file:
cd "C:\Users\PC\source\repos\CudaRuntime4\CudaRuntime4" (the file location)

-----------------------------------------------------------------------------------------------------

2.Compile:
nvcc -o ring_remover.exe kernel.cu tiff_io.cpp -I"C:/Users/PC/Downloads/opencv/build/include" -L"C:/Users/PC/Downloads/opencv/build/x64/vc16/lib" -lopencv_world4100 -lcudart

-----------------------------------------------------------------------------------------------------

3.python run matplot:
python run_computation.py

-----------------------------------------------------------------------------------------------------
4. python run verification (image compare):
python image_compare.py

-----------------------------------------------------------------------------------------------------

Test Output Individually:
ring_remover.exe . . rec_ test_out 1 1 0 0 30 -300 300 300 30 1
ring_remover.exe . . rec_ test_out 2 2 0 0 30 -300 300 300 30 1
….
ring_remover.exe . . rec_ test_out 16 16 0 0 30 -300 300 300 30 1
-----------------------------------------------------------------------------------------------------

5. python plot graph(make sure each result txt file is saved):
python plot.py

-----------------------------------------------------------------------------------------------------
