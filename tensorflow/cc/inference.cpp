//
// Created by David Hattery on 5/3/19.
// Takes tensorflow model and images and returns image classification tiff image
// with name as specified in input and which has first three channels as RGB
// and appends one channel for each class in the model.
// Also scales output tiff size according to optional input percentage.
//

#include <tiffio.h>
#include <assert.h>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/contrib/image/kernels/image_ops.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/graph_transforms/file_utils.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"


// These are all common classes it's handy to reference with no namespace.
using ::tensorflow::Flag;
using ::tensorflow::Tensor;
using ::tensorflow::Status;
using ::tensorflow::string;
using ::tensorflow::int32;
using ::tensorflow::GraphDef;
using ::tensorflow::Scope;
using ::tensorflow::Session;
using ::tensorflow::SessionOptions;

namespace hive_segmentation {

Status ResizeTensor(const Tensor &in_tensor, std::vector<Tensor> *out_tensors, const int output_height, const int output_width){
  auto root = Scope::NewRootScope();
  auto input_tensor = tensorflow::ops::Const(root, in_tensor);
  auto resized = tensorflow::ops::ResizeBilinear(root.WithOpName("resized"), input_tensor, {output_height, output_width});
  tensorflow::ClientSession session(root);
  TF_RETURN_IF_ERROR(session.Run({resized}, out_tensors));
  return Status::OK();
}

Status MinTensor(const Tensor &in_tensor, int output_classes, float &min_class){
  auto root = Scope::NewRootScope();
  std::vector<Tensor> min_tensors;
  auto input_tensor = tensorflow::ops::Const(root, in_tensor);
  auto min = tensorflow::ops::Min(root.WithOpName("min"), input_tensor, {0,1,2});
  tensorflow::ClientSession session(root);
  TF_RETURN_IF_ERROR(session.Run({min}, &min_tensors));
  auto min_class_vals = min_tensors[0].flat_inner_dims<float>();
  min_class = *std::min_element(min_class_vals.data(), min_class_vals.data() + output_classes);
//  std::cout << "Mat mins debug " << min_tensors[0].DebugString() << std::endl;
//  std::cout << "Mat mins " << min_class_vals << std::endl;
//  std::cout << "The mins " << min_class << std::endl;
  return Status::OK();
}

Status MaxTensor(const Tensor &in_tensor, int output_classes, float &max_class){
  auto root = Scope::NewRootScope();
  std::vector<Tensor> max_tensors;
  auto input_tensor = tensorflow::ops::Const(root, in_tensor);
  auto max = tensorflow::ops::Max(root.WithOpName("max"), input_tensor, {0,1,2});
  tensorflow::ClientSession session(root);
  TF_RETURN_IF_ERROR(session.Run({max}, &max_tensors));
  auto max_class_vals = max_tensors[0].flat_inner_dims<float>();
  max_class = *std::min_element(max_class_vals.data(), max_class_vals.data() + output_classes);
//  std::cout << "Mat mins debug " << max_tensors[0].DebugString() << std::endl;
//  std::cout << "Mat mins " << max_class_vals << std::endl;
//  std::cout << "The mins " << max_class << std::endl;
  return Status::OK();
}

Status ParseGraph(const GraphDef *graph_def, string &input_layer, string &output_layer,
                  int32 *input_batch_size, int32 *input_width, int32 *input_height, int32 *input_channels){
//  int32 input_channels = 0;
  std::vector<const tensorflow::NodeDef*> placeholders;
  std::vector<const tensorflow::NodeDef*> variables;
  for (const tensorflow::NodeDef& node : graph_def->node()) {
    if (node.op() == "Placeholder") {
      placeholders.push_back(&node);
    }
    if (node.op() == "Variable" || node.op() == "VariableV2") {
      variables.push_back(&node);
    }
  }

  if (placeholders.empty()) {
    std::cout << "No inputs spotted." << std::endl;
    return tensorflow::errors::OutOfRange("No inputs");
  } else {
//    std::cout << "Found " << placeholders.size() << " possible inputs: ";
    if(placeholders.size() != 1){
      return tensorflow::errors::OutOfRange("Too many inputs to choose from.");
    }
    for (const tensorflow::NodeDef* node : placeholders) {
      string shape_description = "None";
      if (node->attr().count("shape")) {
        tensorflow::TensorShapeProto shape_proto = node->attr().at("shape").shape();
        Status shape_status = tensorflow::PartialTensorShape::IsValidShape(shape_proto);
        if (shape_status.ok()) {
          std::cout << "Input node name " << node->name() << std::endl;
          input_layer = node->name();
          shape_description = tensorflow::PartialTensorShape(shape_proto).DebugString();
//          std::cout << "Tensor shape " << tensorflow::PartialTensorShape(shape_proto) << std::endl;
//          std::cout << "Tensor dimensions " << tensorflow::PartialTensorShape(shape_proto).dims() << std::endl;
//          std::cout << "Input batch size " << tensorflow::PartialTensorShape(shape_proto).dim_size(0) << std::endl;
          *input_batch_size = int32(tensorflow::PartialTensorShape(shape_proto).dim_size(0));
//          std::cout << "Image x size " << tensorflow::PartialTensorShape(shape_proto).dim_size(1) << std::endl;
          *input_width = int32(tensorflow::PartialTensorShape(shape_proto).dim_size(1));
//          std::cout << "Image y size " << tensorflow::PartialTensorShape(shape_proto).dim_size(2) << std::endl;
          *input_height = int32(tensorflow::PartialTensorShape(shape_proto).dim_size(2));
//          std::cout << "Image channels " << tensorflow::PartialTensorShape(shape_proto).dim_size(3) << std::endl;
          *input_channels = int32(tensorflow::PartialTensorShape(shape_proto).dim_size(3));
        } else {
          shape_description = shape_status.error_message();
          return tensorflow::errors::OutOfRange(shape_status.error_message());
        }
      } else {
        return tensorflow::errors::OutOfRange("Unknown shape size.");
      }
    }
//    std::cout << std::endl;
  }
  if(int32(*input_channels) != 3) {
    return tensorflow::errors::OutOfRange("Model graph does not have 3 input channels");
  }

  std::map<string, std::vector<const tensorflow::NodeDef*>> output_map;
  tensorflow::graph_transforms::MapNodesToOutputs(GraphDef(*graph_def), &output_map);
  std::vector<const tensorflow::NodeDef*> output_nodes;
  std::unordered_set<string> unlikely_output_types = {"Const", "Assign", "NoOp", "Placeholder"};
  for (const tensorflow::NodeDef& node : graph_def->node()) {
    if ((output_map.count(node.name()) == 0) &&
        (unlikely_output_types.count(node.op()) == 0)) {
      output_nodes.push_back(&node);
    }
  }

  if (output_nodes.empty()) {
    std::cout << "No outputs spotted." << std::endl;
    return tensorflow::errors::OutOfRange("No outputs");
  } else {
//    std::cout << "Found " << output_nodes.size() << " possible outputs: ";
    if(output_nodes.size() != 1){
      return tensorflow::errors::OutOfRange("Too many outputs to choose from.");
    }
    for (const tensorflow::NodeDef* node : output_nodes) {
//      std::cout << "(name=" << node->name();
//      std::cout << ", op=" << node->op() << ") ";
      output_layer = node->name();
    }
//    std::cout << std::endl;
  }
//  std::cout << std::endl;

  return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string &graph_file_name,
                 std::unique_ptr<Session> *session,
                 string &input_layer, string &output_layer,
                 int32 *input_batch_size, int32 *input_width, int32 *input_height, int32 *input_channels) {
  GraphDef graph_def;
  Status load_graph_status = tensorflow::graph_transforms::LoadTextOrBinaryGraphFile(graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
  }
  Status parse_graph_status = ParseGraph(&graph_def, input_layer, output_layer, input_batch_size, input_width, input_height, input_channels);
  if (!parse_graph_status.ok()) {
    return tensorflow::errors::FailedPrecondition("Failed to parse compute graph at '", graph_file_name, "'");
  }

  session->reset(NewSession(SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

} // end hive_segmentation namespace


int main(int argc, char *argv[]) {
  // These are the command-line flags the program can understand.
  // TODO input should be list of files or a file name containing a list of files and a corresponding file with a list of output file names
  // input should include resize scale percent
  float scale_percent = 100;
  string image_filename = "2016-tesla-model-s-17-of-43.jpg";
  string image_result_filename = "2016-tesla-model-s-17-of-43.tif";
  string graph = "my_model.pb";
  int image_width = 512;
  int image_height = 512;
  int32 input_width = 512;
  int32 input_height = 512;
  int32 input_channels = 3;
  int32 input_batch_size = 1;
  double input_mean = 0;
  double input_std = 255;
  string input_layer = "input_3";
  string output_layer = "bilinear_up_sampling2d_3/ResizeBilinear";
  uint32 output_classes = 0;
  string root_dir = "./";
  // Note that all of these types must be tensorflow types to work with Flag
  std::vector<Flag> flag_list = {
    Flag("image", &image_filename, "full path image to be processed"),
    Flag("results", &image_result_filename, "full path processed image results"),
    Flag("scale", &scale_percent, "percent to scale output results"),
    Flag("graph", &graph, "graph to be executed"),
    Flag("root_dir", &root_dir, "interpret graph file names relative to this directory"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }


  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Error: Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }


  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  // TODO use gpu for session if available
  std::cout << "Set up session" << std::endl;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = ::hive_segmentation::LoadGraph(graph_path, &session, input_layer, output_layer, &input_batch_size, &input_width, &input_height, &input_channels);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << "Error: " << load_graph_status;
    return -1;
  }
  std::cout << "Model Input: " << input_layer << std::endl;
  std::cout << "Model Output: " << output_layer << std::endl;
  std::cout << "Model x: " << input_width << std::endl;
  std::cout << "Model y: " << input_height << std::endl;
  std::cout << "Model c: " << input_channels << std::endl;


  // TODO repeat the following for all images in scene
  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::cout << "Get image '" << image_filename << "' from disk as float array" << std::endl;
  ::cv::Mat orig_image_mat;
  orig_image_mat = ::cv::imread(image_filename, CV_LOAD_IMAGE_COLOR); // newer opencv versions will use IMREAD_COLOR);   // Read the file
  if(! orig_image_mat.data )                              // Check for invalid input
  {
    LOG(ERROR) <<  "Error: Could not open or find the image" << image_filename;
    return -1;
  }
  // Get actual image size so we can rescale results at end with reference to this
  image_width = orig_image_mat.cols;
  image_height = orig_image_mat.rows;
  std::cout << "Image x width " << image_width << " and y height " << image_height << std::endl;
  // Get image mean and stddev for tensorflow normalization
  cv::Scalar mean, stddev;
  cv::meanStdDev(orig_image_mat, mean, stddev);
  input_mean = 0;
  input_std = 0;
  for (int i=0; i<input_channels; i++){
    input_mean += mean[i];
    input_std += stddev[i];
  }
  input_mean /= input_channels;
  input_std /= input_channels;
  std::cout << "Mean all-channel Mean " << input_mean << " and mean all-channel StdDev " << input_std << std::endl;


  // break into pieces if input image is not square
  std::vector<cv::Mat> sub_images;
  cv::Mat leftImage(image_height, image_height, CV_8UC3);
  cv::Mat rightImage(image_height, image_height, CV_8UC3);
  cv::Rect myROI;
  if ( image_height > image_width ) {
    LOG(ERROR) << "Error: Image height is greater than image width";
    return -1;
  }
  if(image_width != image_height) {
    if ( image_width > ( 2 * image_height )){
      LOG(ERROR) << "Error: Insufficient overlap for a two image method--image width is greater than twice height";
      return -1;
    }
    // Setup a rectangle to define square sub-region on left side of image
    myROI = cv::Rect(0, 0, image_height, image_height);
    //    std::cout << "Left " << myROI << std::endl;

    // Crop the full image to that image contained by the rectangle myROI
    // Note that this doesn't copy the data
    leftImage = orig_image_mat(myROI);
    sub_images.push_back(leftImage);
    // Setup a rectangle to define square sub-region on right side of image
    myROI = cv::Rect(image_width - image_height, 0, image_height, image_height);
    //    std::cout << "Right " << myROI << std::endl;
    rightImage = orig_image_mat(myROI);
    sub_images.push_back(rightImage);
  } else { //only have one image to process and no recombining
    sub_images.push_back(orig_image_mat);
  }


  // create tensorflow tensor directly from in-memory opencv mat
  std::vector<Tensor> resized_tensors;
  cv::Mat new_mat;
  cv::resize(orig_image_mat, new_mat, cv::Size(input_width, input_height));
  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({int(sub_images.size()), input_height, input_width, input_channels}));
  auto input_tensor_mapped = input_tensor.tensor<float, 4>();
  for( auto sub_index = 0; sub_index < sub_images.size(); sub_index++) {
    std::cout << "Making tensor for sub image " << sub_index << " of " << sub_images.size() << std::endl;
    for (int y = 0; y < new_mat.rows; y++) {
      for (int x = 0; x < new_mat.cols; x++) {
        cv::Vec3b pixel = new_mat.at<cv::Vec3b>(y, x);
        input_tensor_mapped(sub_index, y, x, 0) = pixel.val[2]; //R
        input_tensor_mapped(sub_index, y, x, 1) = pixel.val[1]; //G
        input_tensor_mapped(sub_index, y, x, 2) = pixel.val[0]; //B
      }
    }
  }


  // Actually run the images through the model.
  std::cout << "Running images in the model" << std::endl;
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, input_tensor}},
                                   {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Error: Running model failed: " << run_status;
    return -1;
  }
  std::cout << "Finished running model with " << outputs.size() << " results" << std::endl;
  if ( outputs.size() != 1 ) {
    LOG(ERROR) << "Error: invalid number of outputs: " << outputs.size();
    return -1;
  }
  auto const &output = outputs[0];
  output_classes = uint(output.shape().dim_size(3));
  std::cout << "Output shape is " << output.shape() << " with " << output.shape().dims() << " dimensions and " << output_classes << " classes" << std::endl;


  // resize model output as percent of actual image dimensions
  auto final_image_height = uint32(scale_percent * double(image_height) / 100);
  auto final_image_width = uint32(scale_percent * double(image_width) / 100);
  std::vector<Tensor> resized_outputs;  Status resize_status = ::hive_segmentation::ResizeTensor(output, &resized_outputs, final_image_height, final_image_height);
  if (!resize_status.ok()) {
    LOG(ERROR) << "Error: Resizing output from model failed: " << resize_status;
    return -1;
  }
  std::cout << "Model results resized to " << (resized_outputs[0]).shape() << " for output" << std::endl;
  // resize original image to use for rgb colors (first three channels) in output tiff file
  cv::Mat resized_mat(final_image_width, final_image_height, CV_8UC3);
  if (scale_percent != 100) {
    cv::resize(orig_image_mat, resized_mat, cv::Size(final_image_width, final_image_height));
  } else {
    resized_mat = orig_image_mat;
  }
  //  std::cout << "Merged output Image x width " << resized_mat.cols << std::endl;
  //  std::cout << "Merged output Image y height " << resized_mat.rows << std::endl;
  assert(resized_mat.cols == final_image_width);
  assert(resized_mat.rows == final_image_height);


  // normalize segmentation data--globally because some classes may not be present in output and relative values between classes should be maintained
  float min_class, max_class;
  Status min_status = ::hive_segmentation::MinTensor(resized_outputs[0], output_classes, min_class);
  if (!min_status.ok()) {
    LOG(ERROR) << "Error: Getting min_class for normalization failed: " << min_status;
    return -1;
  }
  Status max_status = ::hive_segmentation::MaxTensor(resized_outputs[0], output_classes, max_class);
  if (!max_status.ok()) {
    LOG(ERROR) << "Error: Getting max_class for normalization failed: " << max_status;
    return -1;
  }
  auto range_class = max_class - min_class;
  std::cout << "For global normalization, the min class value is " << min_class << " and the max is " << max_class << " with a range of " << range_class << std::endl;


  // prepare output model data for merging into tiff output
  auto tensor_resized_output_map = (resized_outputs[0]).tensor<float, 4>();
  // get the underlying array
  auto resized_output_array = tensor_resized_output_map.data();
  auto *float_resized_output_array = static_cast<float*>(resized_output_array);
  // Make blending array for combining multiple class segmentations
  float blending_factor[final_image_width];
  int overlap = (2 * final_image_height) - final_image_width;
  int offset = final_image_width - final_image_height;
  for (int i = 0; i < final_image_width; i++) {
    if ( sub_images.size() == 1 ) { // only one output so no blending will be done
      blending_factor[i] = 0;
    } else {
      if (i <= overlap) { // only first sub image will be used
        blending_factor[i] = 0;
      } else if (i > final_image_height) { // only second sub image will be used
        blending_factor[i] = 1;
      } else { // two sub images will be combined by a linear scale
        blending_factor[i] = float(i - offset) / float(overlap);
//        std::cout << i << " is " << blending_factor[i] << std::endl;
      }
    }
  }


  // merge and save data into tiff file
  auto output_channels = output_classes + input_channels;
  std::cout << "Building TIFF of size " << final_image_width << " X " << final_image_height << " X " << output_channels << std::endl;
  std::cout << "Saving Tiff to " << image_result_filename << std::endl;
  TIFF *tif = TIFFOpen(image_result_filename.c_str(), "w");
  TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, final_image_width);
  TIFFSetField(tif, TIFFTAG_IMAGELENGTH, final_image_height);
  TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, output_channels);
  TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  //  TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_SEPARATE);
  //  TIFFSetField(tif, TIFFTAG_XRESOLUTION, 1.0);
  //  TIFFSetField(tif, TIFFTAG_YRESOLUTION, 1.0);
  //  TIFFSetField(tif, TIFFTAG_RESOLUTIONUNIT, 1);
  TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
  TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  //  TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 0);
  //  TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, final_image_width));
  //  TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, final_image_width * output_channels));

  uint8_t arr[final_image_width * output_channels];
  for (auto i = 0; i < final_image_height; i++)  {//test with square
    float scale_pixel[3] = {1.f, 1.f, 1.f};
    cv::Vec3b pixel_value;
    float segmentation_value;
    for (auto j = 0; j < final_image_width; j++) {
      pixel_value = resized_mat.at<cv::Vec3b>(i, j);
      for (auto k = 0; k < 3; k++) {
        // set to rgb colors here
        auto scaled_pixel = uint8_t(scale_pixel[k] * float(pixel_value.val[k]));
//        std::cout << "i " << i << " j " << j << " val " << uint(scaled_pixel) << std::endl;
        arr[j*output_channels + k] = scaled_pixel;
      }
      for (int s = 3; s < output_channels; s++) {
        // set to segmentation here and blend between images
        int batch_image;
        if ( blending_factor[j] == 0 ) { // use first sub image
          batch_image = 0;
          segmentation_value =
              float_resized_output_array[(batch_image * final_image_height * final_image_width * input_channels)
                  + (i * final_image_width + j) * input_channels + s - 3];
        } else if ( blending_factor[j] == 1 ) { // use second sub image
          batch_image = 1;
          segmentation_value =
              float_resized_output_array[(batch_image * final_image_height * final_image_width * input_channels)
                  + (i * final_image_width + j) * input_channels + s - 3];
        } else { // use both sub images in proportion using blending factor
          batch_image = 0;
          segmentation_value = (1 - blending_factor[j]) *
              float_resized_output_array[(batch_image * final_image_height * final_image_width * input_channels)
                  + (i * final_image_width + j) * input_channels + s - 3];
          batch_image = 1;
          segmentation_value += blending_factor[j] *
              float_resized_output_array[(batch_image * final_image_height * final_image_width * input_channels)
                  + (i * final_image_width + j) * input_channels + s - 3];
        }
        // normalize it
        segmentation_value -= min_class;
        segmentation_value /= range_class;
        assert(segmentation_value >= 0);
        assert(segmentation_value <= 1.f);
        // scale to 8 bit pixel value
        segmentation_value *= 255.f;
//        std::cout << "pixel " << i << " " << j << " " << s << " is " << segmentation_value << std::endl;
        arr[j*output_channels + s] = uint8_t(segmentation_value);
      }
    }
    TIFFWriteScanline(tif, &arr, i, 0);
  }
  TIFFClose(tif);


  std::cout << "Done" << std::endl;
  return 0;
}