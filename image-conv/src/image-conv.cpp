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
#include "Timer.h"

static const char* inputImagePath = "./Images/cat.bmp";

static float gaussianBlurFilterFactor = 273.0f;
static float gaussianBlurFilter[25] = {
   1.0f,  4.0f,  7.0f,  4.0f, 1.0f,
   4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
   7.0f, 26.0f, 41.0f, 26.0f, 7.0f,
   4.0f, 16.0f, 26.0f, 16.0f, 4.0f,
   1.0f,  4.0f,  7.0f,  4.0f, 1.0f};
static const int gaussianBlurFilterWidth = 5;

static float edgeSobelHorizontalFactor = 1.0f;
static float edgeSobelHorizontal[9] = {
    1.0f,  1.0f, 1.0f,
    0.0f,  0.0f, 0.0f,
    1.0f,  1.0f, 1.0f};
static const int edgeSobelHorizontalWidth = 3;

static float edgeSobelVerticalFactor = 1.0f;
static float edgeSobelVertical[9] = {
    1.0f,  0.0f, 1.0f,
    1.0f,  0.0f, 1.0f,
    1.0f,  0.0f, 1.0f};
static const int edgeSobelVerticalWidth = 3;

enum filterList
{
    GAUSSIAN_BLUR,
    SOBEl_HORIZONTAL,
    SOBEL_VERTICAL,
};

static const int filterSelection = SOBEL_VERTICAL;

#define IMAGE_SIZE (602*380)
constexpr size_t array_size = IMAGE_SIZE;
typedef std::array<float, array_size> FloatArray;

//************************************
// Image Convolution in DPC++ on device: 
//************************************
void ImageConv_v1(queue &q, unsigned char *image_in, char *image_out, float *filter_in, 
    const size_t FilterWidth, const size_t ImageRows, const size_t ImageCols, const size_t Channels) 
{
    buffer<unsigned char, 1> image_in_buf(image_in, range<1>(ImageRows*ImageCols*Channels));
    buffer<char, 1> image_out_buf(image_out, range<1>(ImageRows*ImageCols*Channels));

    range<2> pixelsRange{ImageRows, ImageCols};

    buffer<char, 1> filter_buf(filter_in, range<1>(FilterWidth*FilterWidth));

    /* Compute the filter width (intentionally truncate) */
    int halfFilterWidth = (int)FilterWidth/2;

    // Submit a command group to the queue by a lambda function that contains the
    // data access permission and device computation (kernel).
    q.submit([&](handler &h) {
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

        char sum = 0;
		char sum2 = 0;
		char sum3 = 0;

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
			  
              sum += srcPtr[r*ImageCols+c*Channels] *
                    f_acc[(k+halfFilterWidth)*FilterWidth + 
                        (l+halfFilterWidth)];
			  sum2 += srcPtr[r*ImageCols+(c*Channels)+1] *
                    f_acc[(k+halfFilterWidth)*FilterWidth + 
                        (l+halfFilterWidth)];
			  sum3 += srcPtr[r*ImageCols+(c*Channels)+2] *
                    f_acc[(k+halfFilterWidth)*FilterWidth + 
                        (l+halfFilterWidth)];
          }
        }
         
        /* Write the new pixel value */
        dstPtr[row*ImageCols+col*Channels] = sum;
		dstPtr[row*ImageCols+col*Channels+1] = sum;
		dstPtr[row*ImageCols+col*Channels+2] = sum;
      } 
    );
  });
}


int main() {
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA || FPGA_PROFILE
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
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
  // loop through the platforms to poke into
  for (auto &onePlatform : myPlatforms) {
    std::cout << ++number << " found .." << std::endl << "Platform: " 
    << onePlatform.get_info<info::platform::name>() <<std::endl;
    // loop through the devices
    auto myDevices = onePlatform.get_devices();
    for (auto &oneDevice : myDevices) {
      std::cout << "Device: " 
      << oneDevice.get_info<info::device::name>() <<std::endl;
    }
  }
  std::cout<<std::endl;
#endif

  // set conv filter
  switch (filterSelection)
  {
    case GAUSSIAN_BLUR:
      filterWidth = gaussianBlurFilterWidth;
      filterFactor = gaussianBlurFilterFactor;
      filter = gaussianBlurFilter;
      break;
	case SOBEL_VERTICAL:
	  filterWidth = edgeSobelVerticalWidth;
	  filterFactor = edgeSobelVerticalFactor;
	  filter = edgeSobelVertical;
    default:
      printf("Invalid filter selection.\n");
      return 1;
  }

  for (int i = 0; i < filterWidth*filterWidth; i++)
  {
    filter[i] = filter[i]/filterFactor;
  }

  /* Read in the BMP image */
  
  hInputImage = readBmpFloat(inputImagePath, &imageRows, &imageCols);
  printf("imageRows=%d, imageCols=%d\n", imageRows, imageCols);
  printf("filterWidth=%d, \n", filterWidth);
  /* Allocate space for the output image */


  Timer t;

  try {
    queue q(d_selector, dpc_common::exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

	int width, height, channels;
	unsigned char *img = stbi_load("./Images/dog1.jpg", &width, &height, &channels, 0);
	if(img == NULL) {
	printf("Error in loading the image\n");
	exit(1);
	}
printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);
  hOutputImage = (char *)malloc( imageRows*imageCols * channels * sizeof(char) );
  for(i=0; i<imageRows*imageCols*channels; i++)
    hOutputImage[i] = 0;
    // Image convolution in DPC++
    ImageConv_v1(q, img, hOutputImage, filter, filterWidth, height, width, channels);
	
	stbi_write_png("sky.png", width, height, channels, hOutputImage, width * channels);    
  } catch (exception const &e) {
    std::cout << "An exception is caught for image convolution.\n";
    std::terminate();
  }

  std::cout << t.elapsed().count() << " seconds\n";

  /* Save the output bmp */
  printf("Output image saved as: cat-filtered.bmp\n");
  //writeBmpFloat(hOutputImage, "cat-filtered.bmp", imageRows, imageCols,
   //       inputImagePath);

#ifndef FPGA_PROFILE
  /* Verify result */
  float *refOutput = convolutionGoldFloat(hInputImage, imageRows, imageCols,
    filter, filterWidth);

  writeBmpFloat(refOutput, "cat-filtered-ref.bmp", imageRows, imageCols,
          inputImagePath);

  bool passed = true;
  for (i = 0; i < imageRows*imageCols; i++) {
    if (fabsf(refOutput[i]-hOutputImage[i]) > 0.001f) {
        printf("%f %c\n", refOutput[i], hOutputImage[i]);
        passed = false;
    }
  }
  if (passed) {
    printf("Passed!\n");
    std::cout << "Image Convolution successfully completed on device.\n";
  }
  else {
    printf("Failed!\n");
  }
#endif

  return 0;
}
