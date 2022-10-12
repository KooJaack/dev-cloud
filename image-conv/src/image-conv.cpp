#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include "dpc_common.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if FPGA || FPGA_EMULATOR || FPGA_PROFILE
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

#include "utils.h"
#include "bmp-utils.h"
#include "gold.h"
#include "Filters.h"

static float edgeSobelVerticalFactor = 1.0f;
static float edgeSobelVertical[9] = {
    1.0f, 0.0f, 1.0f,
    1.0f, 0.0f, 1.0f,
    1.0f, 0.0f, 1.0f};
static const int edgeSobelVerticalWidth = 3;

enum filterList
{
  SOBEL_VERTICAL,
};

static const int filterSelection = SOBEL_VERTICAL;

void ImageConv_v1(queue &q, unsigned char *image_in, char *image_out, float *filter_in,
                  const size_t FilterWidth, const size_t ImageRows, const size_t ImageCols, const size_t Channels)
{
  buffer<unsigned char, 1> image_in_buf(image_in, range<1>(ImageRows * ImageCols * Channels));
  buffer<char, 1> image_out_buf(image_out, range<1>(ImageRows * ImageCols * Channels));

  range<2> pixelsRange{ImageRows, ImageCols};

  buffer<float, 1> filter_buf(filter_in, range<1>(FilterWidth * FilterWidth));

  int halfFilterWidth = (int)FilterWidth / 2;

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  q.submit([&](handler &h)
           {
      // Create an accessor to buffers with access permission: read, write or
      // read/write. The accessor is a way to access the memory in the buffer.
      accessor srcPtr(image_in_buf, h, read_only);

      // Another way to get access is to call get_access() member function 
      auto dstPtr = image_out_buf.get_access<access::mode::write>(h);

      // create an accessor to the filter
      auto f_acc = filter_buf.get_access<access::mode::read>(h);

      // Use parallel_for to run image convolution in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // DPC++ supports unnamed lambda kernel by default.
      h.parallel_for(pixelsRange, [=](id<2> item) 
      { 

        // get row and col of the pixel assigned to this work item
        int row = item[0];
        int col = item[1];

        // Half the width of the filter is needed for indexing memory later 
        int halfWidth = (int)(FilterWidth/2);

        // Iterator for the filter */
        int filterIdx = 0;

        // Each work-item iterates around its local area based on the
        // size of the filter 

        int sum[3] = {0, 0, 0};

        /* Apply the filter to the neighborhood */
        for (int k = -halfFilterWidth; k <= halfFilterWidth; k++) 
        {
          for (int l = -halfFilterWidth; l <= halfFilterWidth; l++)
          {
              /* Indices used to access the image */
              int r = row+k;
              int c = col+l;
              
              /* Handle out-of-bounds locations by clamping to
              * the border pixel */
              r = (r < 0) ? 0 : r;
              c = (c < 0) ? 0 : c;
              r = (r >= ImageRows) ? ImageRows-1 : r;
              c = (c >= ImageCols) ? ImageCols-1 : c;

              #pragma unroll
              for(int i = 0; i < 3; i++)
              {
                sum[i] += srcPtr[r*ImageCols*Channels+(c*Channels)+i] *
                    f_acc[(k+halfFilterWidth)*FilterWidth + 
                        (l+halfFilterWidth)];
              }
          }
        }
        #pragma unroll
        for(int i = 0; i < 3; i++)
        {
          char x;
          if(sum[i] > 255 || sum[i] < 0)
            x = (char)255;
          else
            x = (char)sum[i];

          dstPtr[row*ImageCols*Channels+col*Channels+i] = x;
        }
      } 
    ); });
}

int main()
{
#if FPGA_EMULATOR
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA || FPGA_PROFILE
  ext::intel::fpga_selector d_selector;
#else
  default_selector d_selector;
#endif

  float *hInputImage;
  char *hOutputImage;

  int imageRows;
  int imageCols;
  int i;

  /* Set the filter here */
  cl_int filterWidth;
  float filterFactor;
  float *filter;

#ifndef FPGA_PROFILE
  // Query about the platform
  unsigned number = 0;
  auto myPlatforms = platform::get_platforms();
#endif

  // set conv filter
  switch (filterSelection)
  {
  case SOBEL_VERTICAL:
    filterWidth = edgeSobelVerticalWidth;
    filterFactor = edgeSobelVerticalFactor;
    filter = edgeSobelVertical;
    break;
  default:
    printf("Invalid filter selection.\n");
    return 1;
  }

  for (int i = 0; i < filterWidth * filterWidth; i++)
  {
    filter[i] = filter[i] / filterFactor;
  }

  try
  {
    queue q(d_selector, dpc_common::exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    int width, height, channels;
    unsigned char *img = stbi_load("./Images/dog1.jpg", &width, &height, &channels, 0);
    if (img == NULL)
    {
      printf("Error in loading the image\n");
      exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);
    stbi_write_jpg("test.jpg", width, height, channels, img, width * channels);
    printf("Reference write image as test.jpg");

    hOutputImage = (char *)malloc(imageRows * imageCols * channels * sizeof(char));
    for (i = 0; i < imageRows * imageCols * channels; i++)
      hOutputImage[i] = 0;

    // Image convolution in DPC++
    ImageConv_v1(q, img, hOutputImage, filter, filterWidth, height, width, channels);

    stbi_write_jpg("dogcringe.jpg", width, height, channels, hOutputImage, width * channels);
  }
  catch (exception const &e)
  {
    std::cout << "An exception is caught for image convolution.\n";
    std::terminate();
  }

  printf("Output image saved as: dogcringe.jpg\n");

  return 0;
}
