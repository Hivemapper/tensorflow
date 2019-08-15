//
// Created by David Hattery on 5/3/19.
// Takes tensorflow model and images and returns image classification tiff image
// with name as specified in input and which has first three channels as RGB
// and appends one channel for each class in the model.
// Also scales output tiff size according to optional input percentage.
//

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iterator>
#include <tiffio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

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


// This needs to be global to prevent problems with eigen
using ::tensorflow::Tensor;

namespace hive_segmentation {

// These are all common classes it's handy to reference with no namespace.
using ::tensorflow::Flag;
using ::tensorflow::GraphDef;
using ::tensorflow::int32;
using ::tensorflow::Scope;
using ::tensorflow::Session;
using ::tensorflow::SessionOptions;
using ::tensorflow::Status;
using ::tensorflow::string;
using ::std::ifstream;
using ::std::istream_iterator;

// This method just resizes a tensors with bicubic interpolation on each class/channel
Status ResizeTensor(const Tensor &in_tensor, std::vector<Tensor> *out_tensors, const int output_height, const int output_width){
  auto root = Scope::NewRootScope();
  auto input_tensor = tensorflow::ops::Const(root, in_tensor);
  // bicubic is more computationally complex but gives smoother results
//  auto resized = tensorflow::ops::ResizeBilinear(root.WithOpName("resized"), input_tensor, {output_height, output_width});
  auto resized = tensorflow::ops::ResizeBicubic(root.WithOpName("resized"), input_tensor, {output_height, output_width});
  tensorflow::ClientSession session(root);
  TF_RETURN_IF_ERROR(session.Run({resized}, out_tensors));
  return Status::OK();
}

// This method is like the per_image_standardization available in python tensorflow.
// It resizes the input and subtracts the global mean and divides by the global standard deviation.
Status NormalizeTensor(const Tensor &in_tensor, std::vector<Tensor> *out_tensors, const int output_height, const int output_width) {
  auto root = Scope::NewRootScope();
  auto input_tensor = tensorflow::ops::Const(root, in_tensor);
  auto resized = tensorflow::ops::ResizeBilinear(root.WithOpName("resized"), input_tensor, {output_height, output_width});
  auto mean = tensorflow::ops::Mean(root.WithOpName("mean"), resized, {0,1,2,3});
  auto mean2 = tensorflow::ops::Mean(root.WithOpName("mean2"), tensorflow::ops::Square(root.WithOpName("square"), resized), {0,1,2,3});
  auto variance = tensorflow::ops::Sub(root.WithOpName("variance"), mean2, tensorflow::ops::Square(root, mean));
  auto standarddev = tensorflow::ops::Sqrt(root.WithOpName("standarddev"), variance);
  // need a min value for standarddev = 1/sqrt(num_pixels)
  // TODO dwh: test if this is number of pixels (probably) or colors * pixels (but OK too)
  // alternative is something like the python call: num_pixels = math_ops.reduce_prod(array_ops.shape(image)[-3:])
  auto num_pixels = tensorflow::ops::Cast(root.WithOpName("num_pixels"), tensorflow::ops::Size(root, resized), tensorflow::DT_FLOAT);
  auto min_std = tensorflow::ops::Rsqrt(root.WithOpName("min_std"),  num_pixels);
  auto adj_std = tensorflow::ops::Maximum(root.WithOpName("adj_std"), standarddev, min_std);
  auto normalized = tensorflow::ops::Div(root.WithOpName("normalized"), tensorflow::ops::Sub(root, resized, mean), adj_std);
  tensorflow::ClientSession session(root);
  TF_RETURN_IF_ERROR(session.Run({normalized}, out_tensors));
  return Status::OK();
}

Status MeanTensor(const Tensor &in_tensor, float &output_mean){
  auto root = Scope::NewRootScope();
  std::vector<Tensor> mean_tensors {};
  auto input_tensor = tensorflow::ops::Const(root, in_tensor);
  auto mean = tensorflow::ops::Mean(root.WithOpName("mean"), input_tensor, {0,1,2,3});
  tensorflow::ClientSession session(root);
  TF_RETURN_IF_ERROR(session.Run({mean}, &mean_tensors));
  auto mean_vals = mean_tensors[0].flat_inner_dims<float>();
//  std::cout << "Mean debug " << mean_tensors[0].DebugString() << std::endl;
//  std::cout << "Mat mins " << min_class_vals << std::endl;
  output_mean = mean_vals.data()[0];
//  std::cout << "The mean " << output_mean << std::endl;
  return Status::OK();
}

Status StdTensor(const Tensor &in_tensor, float input_mean, float &output_std){
  auto root = Scope::NewRootScope();
  std::vector<Tensor> std_tensors {};
  auto input_tensor = tensorflow::ops::Const(root, in_tensor);
  auto mean2 = tensorflow::ops::Mean(root.WithOpName("mean"), tensorflow::ops::Square(root.WithOpName("square"), input_tensor), {0,1,2,3});
  tensorflow::ClientSession session(root);
  TF_RETURN_IF_ERROR(session.Run({mean2}, &std_tensors));
  auto std_vals = std_tensors[0].flat_inner_dims<float>();
//  std::cout << "mean2 debug " << std_tensors[0].DebugString() << std::endl;
//  std::cout << "mean2 " << std_vals << std::endl;
//  std::cout << "The mins " << min_class << std::endl;
  output_std = std::sqrt(std_vals.data()[0] - (input_mean * input_mean));
//  std::cout << "The std " << output_std << std::endl;
  return Status::OK();
}

Status MinTensor(const Tensor &in_tensor, int output_classes, float &min_class){
  auto root = Scope::NewRootScope();
  std::vector<Tensor> min_tensors {};
  auto input_tensor = tensorflow::ops::Const(root, in_tensor);
  auto min_t_class = tensorflow::ops::Min(root.WithOpName("min_t_class"), input_tensor, {0,1,2});
  tensorflow::ClientSession session(root);
  TF_RETURN_IF_ERROR(session.Run({min_t_class}, &min_tensors));
  auto min_class_vals = min_tensors[0].flat_inner_dims<float>();
  min_class = *std::min_element(min_class_vals.data(), min_class_vals.data() + output_classes);
//  std::cout << "Mat mins debug " << min_tensors[0].DebugString() << std::endl;
//  std::cout << "Mat mins " << min_class_vals << std::endl;
//  std::cout << "The mins " << min_class << std::endl;
  return Status::OK();
}

Status MaxTensor(const Tensor &in_tensor, int output_classes, float &max_class){
  auto root = Scope::NewRootScope();
  std::vector<Tensor> max_tensors {};
  auto input_tensor = tensorflow::ops::Const(root, in_tensor);
  auto max_t_class = tensorflow::ops::Max(root.WithOpName("max_t_class"), input_tensor, {0,1,2});
  tensorflow::ClientSession session(root);
  TF_RETURN_IF_ERROR(session.Run({max_t_class}, &max_tensors));
  auto max_class_vals = max_tensors[0].flat_inner_dims<float>();
  max_class = *std::max_element(max_class_vals.data(), max_class_vals.data() + output_classes);
//  std::cout << "Mat maxs debug " << max_tensors[0].DebugString() << std::endl;
//  std::cout << "Mat maxs " << max_class_vals << std::endl;
//  std::cout << "The maxs " << max_class << std::endl;
  return Status::OK();
}

// parses a graph and gets the names of input and output layers, as well as input tensor dimensions
// that can be used to resize input images for processing in the model
Status ParseGraph(const GraphDef *graph_def, string &input_layer, string &output_layer,
                  int32& input_batch_size, int32& input_width, int32& input_height, int32& input_channels){
  std::vector<const tensorflow::NodeDef*> placeholders {};
  std::vector<const tensorflow::NodeDef*> variables {};
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
          input_batch_size = static_cast<int32>(tensorflow::PartialTensorShape(shape_proto).dim_size(0));
//          std::cout << "Image x size " << tensorflow::PartialTensorShape(shape_proto).dim_size(1) << std::endl;
          input_width = static_cast<int32>(tensorflow::PartialTensorShape(shape_proto).dim_size(1));
//          std::cout << "Image y size " << tensorflow::PartialTensorShape(shape_proto).dim_size(2) << std::endl;
          input_height = static_cast<int32>(tensorflow::PartialTensorShape(shape_proto).dim_size(2));
//          std::cout << "Image channels " << tensorflow::PartialTensorShape(shape_proto).dim_size(3) << std::endl;
          input_channels = static_cast<int32>(tensorflow::PartialTensorShape(shape_proto).dim_size(3));
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
  if(static_cast<int32>(input_channels) != 3) {
    return tensorflow::errors::OutOfRange("Model graph does not have 3 input channels");
  }

  std::map<string, std::vector<const tensorflow::NodeDef*>> output_map {};
  tensorflow::graph_transforms::MapNodesToOutputs(GraphDef(*graph_def), &output_map);
  std::vector<const tensorflow::NodeDef*> output_nodes {};
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

// Reads a model graph definition from disk, and creates a session object
Status LoadGraph(const string &graph_file_name,
                 std::unique_ptr<Session> *session,
                 string &input_layer, string &output_layer,
                 int32& input_batch_size, int32& input_width, int32& input_height, int32& input_channels) {
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
  return session_create_status;
}

int load_images_from_file(const string &image_filename,
                          const string &image_result_filename,
                          std::vector<std::string> *bulk_images,
                          std::vector<std::string> *bulk_results) {

  ifstream bulk_image_file(image_filename);
  std::copy(istream_iterator<string>(bulk_image_file),
            istream_iterator<string>(),
            std::back_inserter(*bulk_images));
  if (bulk_images->empty()) {
    LOG(ERROR) <<  "Error: Could not open or find the image: " << image_filename;
    return -1;
  } else {
    for (auto &filestr : *bulk_images) filestr.erase(std::remove(filestr.begin(), filestr.end(), '\"'), filestr.end());
    std::cout << "Read in " << bulk_images->size() << " input images" << std::endl;
  }
  if (!image_result_filename.empty()) {
    // try opening image_result_filename as an image else it is a txt file with strings
    auto orig_image_mat = ::cv::imread(image_result_filename, cv::COLOR_BGR2RGB);
    if(orig_image_mat.data) {
      LOG(ERROR) <<  "Error: input is an image file list but results is an image: " << image_result_filename;
      return -1;
    } else {
      ifstream bulk_results_file(image_result_filename);
      std::copy(istream_iterator<string>(bulk_results_file),
                istream_iterator<string>(),
                std::back_inserter(*bulk_results));
    }
  }
  // if there is no results file, or outputs is not equal to inputs, build outputs based on image_filename file names with tif extension
  if (image_result_filename.empty() || (bulk_images->size() != bulk_results->size())) {
    bulk_results->clear();
    for (auto const &image_file : *bulk_images) {
      bulk_results->push_back(image_file.substr(0, image_file.find_last_of('.'))+".tif");
    }
    std::cout << "Generated " << bulk_results->size() << " result image file names" << std::endl;
  } else {
    for (auto &filestr : *bulk_results) filestr.erase(std::remove(filestr.begin(), filestr.end(), '\"'), filestr.end());
    std::cout << "Read in " << bulk_results->size() << " result image file names" << std::endl;
  }
  return 0;
}

// This creates a linear array that has a flat zone, a linear ramp zone and a final flat zone
// where the ramp is in the area where two images overlap and first and last flat areas are non-overlap areas.
// The values range from 0 to 1 and define the proportions of classes in the two images that will be merged into the final image
float *blender_array(int num_vals, int sub_size) {
  int offset = num_vals - sub_size;
  int overlap = (2 * sub_size) - num_vals;
  assert(overlap >= 0);
  std::cout << "Making blending array of size " << num_vals << ", with offset " << offset << ", overlap " << overlap << " and sub_size " << sub_size << std::endl;
  float *blender = new float[num_vals] {};
  for (std::size_t i=0; i<num_vals; i++) {
    if (i <= offset) {
      // only first sub image will be used
      blender[i] = 0;
    } else if (i > sub_size) {
      // only second sub image will be used
      blender[i] = 1;
    } else {
      // two sub images will be combined by a linear scale
      blender[i] = float(i - offset) / float(overlap);
//      std::cout << i << " is " << blender[i] << std::endl;
    }
  }
  return blender;
}

// Takes a tensorflow result pointer and two indices into that pointer corresponding to the two images to merge horizontally.
// num_y and num_x are the sizes of the images in the tensorflow data and target_y and target_x are the size that
// the combined image should be.
// The merged_output should be a pre-allocated array of the correct size in the calling function.
// Note that because tensorflow data is inverted from normal, this function reverses the indices when creating the merged image
void hz_merge(float *results,
              const int first_image,
              const int second_image,
              const int num_y,
              const int num_x,
              const int num_classes,
              float *merged_output,
              const int target_y,
              const int target_x) {
  assert(num_y == target_y);
  assert(results != nullptr);
  assert(merged_output != nullptr);
  int offset = target_x - num_x;
  auto x_blending_factor = blender_array(target_x, num_x);

  std::cout << "Performing the merge" << std::endl;
  for (auto i = 0; i < target_x; i++) {
    for (auto j = 0; j < target_y; j++) {
      for (auto s = 0; s < num_classes; s++) {
        float segmentation_value = 0;
        if (x_blending_factor[i] == 0) {
          // use first sub image
          // note tensorflow data is reversed cols and rows
          segmentation_value = results[(first_image * num_y * num_x * num_classes) + (j * num_y + i) * num_classes + s];
        } else if (x_blending_factor[i] == 1) {
          // use second sub image
          segmentation_value = results[(second_image * num_y * num_x * num_classes) + (j * num_y + (i - offset)) * num_classes + s];
        } else {
          // use both sub images in proportion using blending factor
          segmentation_value = (1 - x_blending_factor[i]) * results[(first_image * num_y * num_x * num_classes) + (j * num_y + i) * num_classes + s];
          segmentation_value += x_blending_factor[i] * results[(second_image * num_y * num_x * num_classes) + (j * num_y + (i - offset)) * num_classes + s];
        }
//        std::cout << "pixel " << i << " " << j << " " << s << " is " << segmentation_value << std::endl;
        merged_output[(i * target_y + j) * num_classes + s] = segmentation_value;
      }
    }
  }
}

// Takes two image pointers corresponding to two images to merge horizontally.
// num_y and num_x are the sizes of the source images and target_y and target_x are the size that
// the combined image should be.
// The merged_output should be a pre-allocated array of the correct size in the calling function.
// Note this function does not reverse the indices when creating the merged image
void hz_merge(const float *first_image,
              const float *second_image,
              const int num_y,
              const int num_x,
              const int num_classes,
              float *merged_output,
              const int target_y,
              const int target_x) {
  assert(num_y == target_y);
  assert(first_image != nullptr);
  assert(second_image != nullptr);
  assert(merged_output != nullptr);
  int offset = target_x - num_x;
  auto x_blending_factor = blender_array(target_x, num_x);

  std::cout << "Performing the merge" << std::endl;
  for (auto i = 0; i < target_x; i++) {
    for (auto j = 0; j < target_y; j++) {
      for (auto s = 0; s < num_classes; s++) {
        float segmentation_value = 0;
        if (x_blending_factor[i] == 0) {
          // use first sub image
          segmentation_value = first_image[(i * num_y + j) * num_classes + s];
        } else if (x_blending_factor[i] == 1) {
          // use second sub image
          segmentation_value = second_image[((i - offset) * num_y + j) * num_classes + s];
        } else {
          // use both sub images in proportion using blending factor
          segmentation_value = (1 - x_blending_factor[i]) * first_image[(i * num_y + j) * num_classes + s];
          segmentation_value += x_blending_factor[i] * second_image[((i - offset) * num_y + j) * num_classes + s];
        }
//        std::cout << "pixel " << i << " " << j << " " << s << " is " << segmentation_value << std::endl;
        merged_output[(i * target_y + j) * num_classes + s] = segmentation_value;
      }
    }
  }
}

// Takes a tensorflow result pointer and two indices into that pointer corresponding to the two images to merge vertically.
// num_y and num_x are the sizes of the images in the tensorflow data and target_y and target_x are the size that
// the combined image should be.
// The merged_output should be a pre-allocated array of the correct size in the calling function.
// Note that because tensorflow data is inverted from normal, this function reverses the indices when creating the merged image
void vert_merge(float *results,
                const int first_image,
                const int second_image,
                const int num_y,
                const int num_x,
                const int num_classes,
                float *merged_output,
                const int target_y,
                const int target_x) {
  assert(num_x == target_x);
  assert(results != nullptr);
  assert(merged_output != nullptr);
  int offset = target_y - num_y;
  auto y_blending_factor = blender_array(target_y, num_y);

  std::cout << "Performing the merge" << std::endl;
  for (auto i = 0; i < target_x; i++) {
    for (auto j = 0; j < target_y; j++) {
      for (auto s = 0; s < num_classes; s++) {
        float segmentation_value = 0;
        if (y_blending_factor[j] == 0) {
          // use first sub image
          // note tensorflow data is reversed cols and rows
          segmentation_value = results[(first_image * num_y * num_x * num_classes) + (j * num_y + i) * num_classes + s];
        } else if (y_blending_factor[j] == 1) {
          // use second sub image
          segmentation_value = results[(second_image * num_y * num_x * num_classes) + ((j - offset) * num_y + i) * num_classes + s];
        } else {
          // use both sub images in proportion using blending factor
          segmentation_value = (1 - y_blending_factor[j]) * results[(first_image * num_y * num_x * num_classes) + (j * num_y + i) * num_classes + s];
          segmentation_value += y_blending_factor[j] * results[(second_image * num_y * num_x * num_classes) + ((j - offset) * num_y + i) * num_classes + s];
        }
        merged_output[(i * target_y + j) * num_classes + s] = segmentation_value;
      }
    }
  }
}

// Takes two image pointers corresponding to two images to merge vertically.
// num_y and num_x are the sizes of the source images and target_y and target_x are the size that
// the combined image should be.
// The merged_output should be a pre-allocated array of the correct size in the calling function.
// Note this function does not reverse the indices when creating the merged image
void vert_merge(const float *first_image,
                const float *second_image,
                const int num_y,
                const int num_x,
                const int num_classes,
                float *merged_output,
                const int target_y,
                const int target_x) {
  assert(num_x == target_x);
  assert(first_image != nullptr);
  assert(second_image != nullptr);
  assert(merged_output != nullptr);
  int offset = target_y - num_y;
  auto y_blending_factor = blender_array(target_y, num_y);

  std::cout << "Performing the merge" << std::endl;
  for (auto i = 0; i < target_x; i++) {
    for (auto j = 0; j < target_y; j++) {
      for (auto s = 0; s < num_classes; s++) {
        float segmentation_value = 0;
        if (y_blending_factor[j] == 0) {
          // use first sub image
          segmentation_value = first_image[(i * num_y + j) * num_classes + s];
        } else if (y_blending_factor[j] == 1) { // use second sub image
          segmentation_value = second_image[(i * num_y + (j - offset)) * num_classes + s];
        } else {
          // use both sub images in proportion using blending factor
          segmentation_value = (1 - y_blending_factor[j]) * first_image[(i * num_y + j) * num_classes + s];
          segmentation_value += y_blending_factor[j] * second_image[(i * num_y + (j - offset)) * num_classes + s];
        }
        merged_output[(i * target_y + j) * num_classes + s] = segmentation_value;
      }
    }
  }
}

// This takes two pointers to images and adds them together.  The two images must be the same size--num_y by num_x
// and the output will be the same size as the input images.
// The merged_output should be a pre-allocated array of the correct size in the calling function.
void add_images(const float *first_image,
                const float *second_image,
                const int num_y,
                const int num_x,
                const int num_classes,
                float *merged_output) {
  assert(first_image != nullptr);
  assert(second_image != nullptr);
  assert(merged_output != nullptr);
  std::cout << "Performing the add" << std::endl;
  for (auto i = 0; i < num_x; i++) {
    for (auto j = 0; j < num_y; j++) {
      for (auto s = 0; s < num_classes; s++) {
        float segmentation_value = 0;
        // use both sub images in equal proportion
        segmentation_value = 0.5f * first_image[(i * num_y + j) * num_classes + s];
        segmentation_value += 0.5f * second_image[(i * num_y + j) * num_classes + s];
        merged_output[(i * num_y + j) * num_classes + s] = segmentation_value;
      }
    }
  }
}

int write_tiff_file(const float *merged_output_classes,
                    ::cv::Mat resized_mat,
                    const int final_image_height,
                    const int final_image_width,
                    const int input_channels,
                    const int output_classes,
                    const string& image_result_filename) {
  auto output_channels = output_classes + input_channels;
  std::cout << "Building TIFF of size " << final_image_width << " X " << final_image_height << " X " << output_channels << std::endl;
  std::cout << "Saving Tiff to " << image_result_filename << std::endl;
  TIFF *tif = TIFFOpen(image_result_filename.c_str(), "w");
  TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, final_image_width);
  TIFFSetField(tif, TIFFTAG_IMAGELENGTH, final_image_height);
  TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, output_channels);
  // if 8-bits doesn't have the fidelity needed can switch to 16 bit tiff images
  TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
  TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
  TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, final_image_height));
  TIFFSetField(tif, TIFFTAG_SUBFILETYPE, 0x0);

  uint8_t arr[final_image_width * output_channels];
  for (std::size_t i=0; i<final_image_height; i++)  {
    float scale_pixel[3] = {1.f, 1.f, 1.f};
    for (std::size_t j=0; j<final_image_width; j++) {
      auto pixel_value = resized_mat.at<cv::Vec3b>(i, j);
      std::vector<double> segmentation_floats {};
      float segmentation_value = 0;
      for (std::size_t k=0; k<3; k++) {
        // set to rgb colors here
        // we need BGR in tiff for hivemapper to extract right colors so reverse pixel color order
        auto scaled_pixel = uint8_t(scale_pixel[k] * static_cast<float>(pixel_value.val[2-k]));
        //        std::cout << "i " << i << " j " << j << " val " << uint(scaled_pixel) << std::endl;
        arr[j*output_channels + k] = scaled_pixel;
      }
      for (int s=0; s<output_classes; s++) {
        segmentation_value = merged_output_classes[(j * final_image_height + i) * output_classes + s];
        segmentation_floats.push_back(segmentation_value);
        //        segmentation_floats->insert(s, segmentation_value);
//          std::cout << "pixel " << i << " " << j << " " << s << " is " << segmentation_value << std::endl;
      }
      // normalize classes to 0-1 float values
      auto normalization_min = *std::min_element(segmentation_floats.begin(), segmentation_floats.end());
      for (auto &value : segmentation_floats) value -= normalization_min;
      auto normalization_max = *std::max_element(segmentation_floats.begin(), segmentation_floats.end());
      auto normalization_range = normalization_max - normalization_min;
      normalization_range = (normalization_range == 0)? 1. : normalization_range;
      //      std::cout << "Range pixel " << i << " " << j << " min is " << normalization_min << " and max is " << normalization_max << " for total range " << normalization_range << std::endl;
      for (auto &value : segmentation_floats) value /= normalization_range;
      // make into probability using sum of all values in pixel classes
      float normalization_sum = 0;
      for (auto value : segmentation_floats) normalization_sum += value;
      // if sum is zero then all classes are equally possible--note argmax takes first match which will be unknown class
      normalization_sum = (normalization_sum == 0)? 1. / static_cast<float>(output_classes) : normalization_sum;
      for (auto &value : segmentation_floats) value /= normalization_sum;
      //      std::cout << "Sum is " << normalization_sum << std::endl;
      //      std::cout << "Normalized pixel " << i << " " << j << " ";
      //      for (auto &value : segmentation_floats) std::cout << value << " ";
      //      std::cout << std::endl;
      for (int s=0; s<output_classes; s++) {
        // scale to 8 bit pixel value--note may not sum to 1 now so not strictly a probability anymore
        arr[j*output_channels + input_channels + s] = static_cast<uint8_t>(std::round(segmentation_floats[s] * 255.f));
        //        std::cout << "Final pixel " << i << " " << j << " " << s << " " << std::round(segmentation_floats[s] * 255.f) << std::endl;
      }
    }
    TIFFWriteScanline(tif, &arr, i, 0);
  }
  TIFFClose(tif);
  return 0;
}





} // end hive_segmentation namespace


int main(int argc, char *argv[]) {

  // These are all common classes it's handy to reference with no namespace.
  using ::tensorflow::Flag;
  using ::tensorflow::GraphDef;
  using ::tensorflow::int32;
  using ::tensorflow::Scope;
  using ::tensorflow::Session;
  using ::tensorflow::SessionOptions;
  using ::tensorflow::Status;
  using ::tensorflow::string;
  using ::std::ifstream;
  using ::std::istream_iterator;

  // These are the command-line flags the program can understand.
  // resize scale percent for output results
  float scale_percent = 100;
  // image_filename can be an image or a file with a list of images--this must be set when invoking function
  string image_filename;
  // optional output result filename defaults to imagename.tif but can be specified as an imagename or a file with a list of imagenames
  string image_result_filename;
  // the tensorflow graph name without the directory (root_dir below is prefixed to name)
  string graph = "segmentation_model.pb";
  string root_dir = "./";

  // some config
  bool do_quads = true;
  float overlap_fraction = 1.1;

  // data structures to hold multiple image and result names in sequence order
  std::vector<std::string> bulk_images {};
  std::vector<std::string> bulk_results {};

  // data with defaults
  int image_width = 512;
  int image_height = 512;
  int32 input_width = 512;
  int32 input_height = 512;
  double input_aspect_ratio = 1.0;
  int32 input_channels = 3;
  int32 input_batch_size = 1;
  string input_layer = "input_3";
  string output_layer = "bilinear_up_sampling2d_3/ResizeBilinear";
  uint32 output_classes = 0;


  // check flags--Note that all of these types must be tensorflow types to work with Flag
  std::vector<Flag> flag_list = {
    Flag("image", &image_filename, "full path image to be processed--no default and mandatory"),
    Flag("results", &image_result_filename, "full path processed image results--default is image filename with .tif extension"),
    Flag("scale", &scale_percent, "percent to scale output results--default is 100"),
    Flag("graph", &graph, "graph to be executed--default is segmentation_model.pb"),
    Flag("root_dir", &root_dir, "interpret graph file names relative to this directory--default is ./"),
    Flag("do_quads", &do_quads, "do quad breakdown in addition to squaring up--default is true"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || image_filename.empty()) {
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
  // TODO dwh: use gpu for session if available
  std::cout << "Set up session" << std::endl;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  Status load_graph_status = ::hive_segmentation::LoadGraph(graph_path, &session, input_layer, output_layer, input_batch_size, input_width, input_height, input_channels);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << "Error: " << load_graph_status;
    return -1;
  }
  input_aspect_ratio = static_cast<double>(input_height) / static_cast<double>(input_width);
  std::cout << "Model Input: " << input_layer << std::endl;
  std::cout << "Model Output: " << output_layer << std::endl;
  std::cout << "Model x: " << input_width << std::endl;
  std::cout << "Model y: " << input_height << std::endl;
  std::cout << "Model aspect ratio: " << input_aspect_ratio << std::endl;
  std::cout << "Model colors: " << input_channels << std::endl;


  // try loading image_filename as an image and if it fails, check to see if it is a list of images
  auto orig_image_mat = ::cv::imread(image_filename, cv::COLOR_BGR2RGB);
  if(!orig_image_mat.data) {                             // Check for invalid input
    // try opening image_filename as a txt file with strings and same with image_result_filename
    int success = hive_segmentation::load_images_from_file(image_filename, image_result_filename, &bulk_images, &bulk_results);
    if (success != 0) {
      return -1;
    }
  } else {
    bulk_images.push_back(image_filename);
    bulk_results.push_back(image_result_filename);
  }


  // now process images one at a time
  while (!bulk_images.empty() && !bulk_results.empty()){
    image_filename = bulk_images.back();
    bulk_images.pop_back();
    image_result_filename = bulk_results.back();
    bulk_results.pop_back();

    // Get the image from disk as a float array of numbers, resized and normalized
    // to the specifications the main graph expects.
    std::cout << "Get image '" << image_filename << "' from disk as float array" << std::endl;
    // note natural imread uses BGR color order so want to use RGB instead
    orig_image_mat = ::cv::imread(image_filename, cv::COLOR_BGR2RGB);// CV_LOAD_IMAGE_COLOR); // newer opencv versions will use IMREAD_COLOR);   // Read the file as RGB
//  orig_image_mat = ::cv::imread(image_filename, cv::IMREAD_COLOR);   // Read the file as BGR
    if(!orig_image_mat.data){
      LOG(ERROR) <<  "Error: Could not open or find the image: " << image_filename;
      return -1;
    }


    // Get actual image size so we can rescale results at end with reference to this
    image_width = orig_image_mat.cols;
    image_height = orig_image_mat.rows;
    std::cout << "Image " << image_filename << " x width " << image_width << " and y height " << image_height << std::endl;


    // break into pieces if input image is not square
    std::vector<cv::Mat> sub_images {};
    std::vector<cv::Rect> rectangles {};
    if (image_height > static_cast<int>(float(image_width) * input_aspect_ratio)) {
      LOG(ERROR) << "Error: Image height is proportionally greater than image width";
      return -1;
    }
    int sub_image_height = image_height;
    int sub_image_width = static_cast<int>(static_cast<float>(image_height) / input_aspect_ratio);
    cv::Mat leftImage(sub_image_width, sub_image_height, CV_8UC3);
    cv::Mat rightImage(sub_image_width, sub_image_height, CV_8UC3);
    if (image_height != static_cast<int>(static_cast<float>(image_width) * input_aspect_ratio)) {
      if (static_cast<double>(image_width) * input_aspect_ratio > static_cast<double>(2 * image_height)){
        LOG(ERROR) << "Error: Insufficient overlap for a two image method--image width is greater than twice height";
        return -1;
      }
      // Setup a rectangle to define square sub-region on left side of image
      auto leftROI = cv::Rect(0, 0, leftImage.cols, leftImage.rows);
      //    std::cout << "Left " << leftROI << std::endl;
      // Crop the full image to that image contained by the rectangle myROI
      // Note that this doesn't copy the data
      leftImage = orig_image_mat(leftROI);

      // Setup a rectangle to define square sub-region on right side of image
      auto rightROI = cv::Rect(image_width - rightImage.cols, 0, rightImage.cols, rightImage.rows);
      //    std::cout << "Right " << rightROI << std::endl;
      rightImage = orig_image_mat(rightROI);

      if (do_quads) {
        sub_image_height = std::max(static_cast<int>(static_cast<float>(image_height) * overlap_fraction / 2.f), input_height);
        sub_image_width = static_cast<int>(static_cast<float>(sub_image_height) / input_aspect_ratio);
        std::cout << "Breaking up image into two quads of subimages with height " << sub_image_height << " and width " << sub_image_width << std::endl;
        // TODO dwh: make roi and then use them to make sub images
        sub_images.push_back(leftImage(cv::Rect(0, 0, sub_image_width, sub_image_height)));
        rectangles.push_back(cv::Rect(0, 0, sub_image_width, sub_image_height));
        sub_images.push_back(leftImage(cv::Rect(leftImage.cols - sub_image_width, 0, sub_image_width, sub_image_height)));
        rectangles.push_back(cv::Rect(leftImage.cols - sub_image_width, 0, sub_image_width, sub_image_height));
        sub_images.push_back(leftImage(cv::Rect(0, leftImage.rows - sub_image_height, sub_image_width, sub_image_height)));
        rectangles.push_back(cv::Rect(0, leftImage.rows - sub_image_height, sub_image_width, sub_image_height));
        sub_images.push_back(leftImage(cv::Rect(leftImage.cols - sub_image_width, leftImage.rows - sub_image_height, sub_image_width, sub_image_height)));
        rectangles.push_back(cv::Rect(leftImage.cols - sub_image_width, leftImage.rows - sub_image_height, sub_image_width, sub_image_height));
        sub_images.push_back(rightImage(cv::Rect(0, 0, sub_image_width, sub_image_height)));
        rectangles.push_back(cv::Rect(0, 0, sub_image_width, sub_image_height));
        sub_images.push_back(rightImage(cv::Rect(rightImage.cols - sub_image_width, 0, sub_image_width, sub_image_height)));
        rectangles.push_back(cv::Rect(rightImage.cols - sub_image_width, 0, sub_image_width, sub_image_height));
        sub_images.push_back(rightImage(cv::Rect(0, rightImage.rows - sub_image_height, sub_image_width, sub_image_height)));
        rectangles.push_back(cv::Rect(0, rightImage.rows - sub_image_height, sub_image_width, sub_image_height));
        sub_images.push_back(rightImage(cv::Rect(rightImage.cols - sub_image_width, rightImage.rows - sub_image_height, sub_image_width, sub_image_height)));
        rectangles.push_back(cv::Rect(rightImage.cols - sub_image_width, rightImage.rows - sub_image_height, sub_image_width, sub_image_height));

        std::cout << "Adding resized left and right subimages" << std::endl;
        cv::Mat resizedLeft(sub_image_width, sub_image_height, CV_8UC3);
        cv::resize(leftImage, resizedLeft, cv::Size(sub_image_width, sub_image_height));
        sub_images.push_back(resizedLeft);
        rectangles.push_back(leftROI);
        cv::Mat resizedRight(sub_image_width, sub_image_height, CV_8UC3);
        cv::resize(rightImage, resizedRight, cv::Size(sub_image_width, sub_image_height));
        sub_images.push_back(resizedRight);
        rectangles.push_back(rightROI);
      } else {
        sub_image_width = sub_image_height;
        std::cout << "Breaking up image into two subimages with height " << sub_image_height << " and width " << sub_image_width << std::endl;
        sub_images.push_back(leftImage);
        rectangles.push_back(leftROI);
        sub_images.push_back(rightImage);
        rectangles.push_back(rightROI);
      }
    } else {
      //only have one image to process and no recombining
      std::cout << "Have single image with height " << sub_image_height << " and width " << sub_image_width << std::endl;
      sub_images.push_back(orig_image_mat);
      rectangles.push_back(cv::Rect(0, 0, image_width, image_height));
    }


    // create tensorflow tensor directly from in-memory opencv mat
    // TODO dwh: is height width order correct??
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({static_cast<int>(sub_images.size()), sub_image_width, sub_image_height, input_channels}));
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();
    for (std::size_t sub_index = 0; sub_index < sub_images.size(); sub_index++) {
      std::cout << "Making tensor for sub image " << sub_index << " of " << sub_images.size() << std::endl;
  //    std::cout << "Rows " << sub_images[sub_index].rows << " and cols " << sub_images[sub_index].cols << std::endl;
      for (int y = 0; y < sub_images[sub_index].rows; y++) {
        for (int x = 0; x < sub_images[sub_index].cols; x++) {
          cv::Vec3b pixel = sub_images[sub_index].at<cv::Vec3b>(y, x);
          // tensorflow reads files as BGR, so need to order/reorder data that way
  //        std::cout << "Pixel " << x << " " << y << " is " << pixel << std::endl;
          input_tensor_mapped(sub_index, y, x, 0) = pixel.val[2]; //B
          input_tensor_mapped(sub_index, y, x, 1) = pixel.val[1]; //G
          input_tensor_mapped(sub_index, y, x, 2) = pixel.val[0]; //R
  //        input_tensor_mapped(sub_index, y, x, 0) = pixel.val[0]; //R
  //        input_tensor_mapped(sub_index, y, x, 1) = pixel.val[1]; //G
  //        input_tensor_mapped(sub_index, y, x, 2) = pixel.val[2]; //B
        }
      }
    }


    // resize and normalize input by mean and std
    std::cout << "Resizing and Normalizing input tensor" << std::endl;
    std::vector<Tensor> resized_normal_tensors {};
    Status resized_normalize_status = ::hive_segmentation::NormalizeTensor(input_tensor, &resized_normal_tensors, input_height, input_width); //, input_mean, input_std);
    if (!resized_normalize_status.ok()) {
      LOG(ERROR) << "Error: Input tensor normalization failed: " << resized_normalize_status;
      return -1;
    }


    // Actually run the images through the model.
    std::cout << "Running images in the model" << std::endl;
    std::vector<Tensor> outputs {};
    Status run_status = session->Run({{input_layer, resized_normal_tensors[0]}},
                                     {output_layer}, {}, &outputs);
    if (!run_status.ok()) {
      LOG(ERROR) << "Error: Running model failed: " << run_status;
      return -1;
    }
    std::cout << "Finished running model with " << outputs.size() << " results" << std::endl;
    if (outputs.size() != 1) {
      LOG(ERROR) << "Error: invalid number of outputs: " << outputs.size();
      return -1;
    }
    auto const &output = outputs[0];
    output_classes = uint(output.shape().dim_size(3));
    std::cout << "Output shape is " << output.shape() << " with " << output.shape().dims() << " dimensions and " << output_classes << " classes" << std::endl;


    // resize and merge to get output size depending on switch/case
    std::cout << "Make output array" << std::endl;
    auto *merged_output_classes = new float[image_height * image_width * output_classes];
    float *dual_output_classes = nullptr;
    Status resize_status;
    std::vector<Tensor> resized_outputs {};
    std::vector<Tensor> full_outputs {};
    switch (sub_images.size()) {
      case 10: {
        // make dual output merged classes for later combining with quads
        dual_output_classes = new float[image_height * image_width * output_classes];
        // don't really need to resize the entire output, but for simplicity we do
        resize_status = ::hive_segmentation::ResizeTensor(output, &full_outputs, leftImage.rows, leftImage.cols);
        if (!resize_status.ok()) {
          LOG(ERROR) << "Error: Resizing dual output from model failed: " << resize_status;
          return -1;
        }
        std::cout << "Model dual results resized to " << (full_outputs[0]).shape() << " for merging" << std::endl;
        auto float_output_array10 = static_cast<float *>(full_outputs[0].flat<float>().data());
        hive_segmentation::hz_merge(float_output_array10,
                                    8,
                                    9,
                                    leftImage.rows,
                                    leftImage.cols,
                                    output_classes,
                                    dual_output_classes,
                                    image_height,
                                    image_width);
        // no break here because want to also run the case 8 when we have 10
      }
      case 8: {
        resize_status = ::hive_segmentation::ResizeTensor(output, &resized_outputs, sub_image_height, sub_image_width);
        if (!resize_status.ok()) {
          LOG(ERROR) << "Error: Resizing quad output from model failed: " << resize_status;
          return -1;
        }
        std::cout << "Model quad results resized to " << (resized_outputs[0]).shape() << " for merging" << std::endl;
        auto float_output_array8 = static_cast<float *>(resized_outputs[0].flat<float>().data());

        // need some temporary data structures
        auto *merged_top_quad = new float[sub_image_height * leftImage.cols * output_classes];
        auto *merged_bottom_quad = new float[sub_image_height * leftImage.cols * output_classes];
        auto *merged_left = new float[leftImage.cols * leftImage.rows * output_classes];
        auto *merged_right = new float[rightImage.cols * rightImage.rows * output_classes];

        // now have 8 dimensional array of size sub_image_height x sub_image_width x num_classes
        // ((0 hz 1) vt (2 hz 3)) hz ((4 hz 5) vt (6 hz 7))
        // (  top    vt  bottom ) hz (  top    vt  bottom )
        //         left           hz           right
        hive_segmentation::hz_merge(float_output_array8,
                                    0,
                                    1,
                                    sub_image_height,
                                    sub_image_width,
                                    output_classes,
                                    merged_top_quad,
                                    sub_image_height,
                                    leftImage.cols);

        hive_segmentation::hz_merge(float_output_array8,
                                    2,
                                    3,
                                    sub_image_height,
                                    sub_image_width,
                                    output_classes,
                                    merged_bottom_quad,
                                    sub_image_height,
                                    leftImage.cols);

        hive_segmentation::vert_merge(merged_top_quad,
                                      merged_bottom_quad,
                                      sub_image_height,
                                      leftImage.cols,
                                      output_classes,
                                      merged_left,
                                      leftImage.rows,
                                      leftImage.cols);

        hive_segmentation::hz_merge(float_output_array8,
                                    4,
                                    5,
                                    sub_image_height,
                                    sub_image_width,
                                    output_classes,
                                    merged_top_quad,
                                    sub_image_height,
                                    leftImage.cols);

        hive_segmentation::hz_merge(float_output_array8,
                                    6,
                                    7,
                                    sub_image_height,
                                    sub_image_width,
                                    output_classes,
                                    merged_bottom_quad,
                                    sub_image_height,
                                    leftImage.cols);

        hive_segmentation::vert_merge(merged_top_quad,
                                      merged_bottom_quad,
                                      sub_image_height,
                                      rightImage.cols,
                                      output_classes,
                                      merged_right,
                                      rightImage.rows,
                                      rightImage.cols);

        // test for 10 classes (quads and dual) and do merge accordingly
        if (dual_output_classes) {
          std::cout << "Combining Dual and Quad classes" << std::endl;
          auto *merged_quad_classes = new float[image_height * image_width * output_classes];
          hive_segmentation::hz_merge(merged_left,
                                      merged_right,
                                      rightImage.rows,
                                      rightImage.cols,
                                      output_classes,
                                      merged_quad_classes,
                                      image_height,
                                      image_width);
          // then combine merged_quad_classes and dual_output_classes
          hive_segmentation::add_images(merged_quad_classes,
                                        dual_output_classes,
                                        image_height,
                                        image_width,
                                        output_classes,
                                        merged_output_classes);
          delete [] merged_quad_classes;
        } else {
          // just merge the left and right from the quads
          hive_segmentation::hz_merge(merged_left,
                                      merged_right,
                                      rightImage.rows,
                                      rightImage.cols,
                                      output_classes,
                                      merged_output_classes,
                                      image_height,
                                      image_width);
        }

        delete[] merged_top_quad;
        delete[] merged_bottom_quad;
        delete[] merged_left;
        delete[] merged_right;
        break;
      }
      case 2: {
        resize_status = ::hive_segmentation::ResizeTensor(output, &resized_outputs, leftImage.rows, leftImage.cols);
        if (!resize_status.ok()) {
          LOG(ERROR) << "Error: Resizing dual output from model failed: " << resize_status;
          return -1;
        }
        std::cout << "Model results resized to " << (resized_outputs[0]).shape() << " for merging" << std::endl;
        auto float_output_array2 = static_cast<float *>(resized_outputs[0].flat<float>().data());
        std::cout << "Merging output classes" << std::endl;
        hive_segmentation::hz_merge(float_output_array2,
                                    0,
                                    1,
                                    leftImage.rows,
                                    leftImage.cols,
                                    output_classes,
                                    merged_output_classes,
                                    image_height,
                                    image_width);
        break;
      }
      case 1: {
        resize_status = ::hive_segmentation::ResizeTensor(output, &resized_outputs, image_height, image_width);
        if (!resize_status.ok()) {
          LOG(ERROR) << "Error: Resizing single output from model failed: " << resize_status;
          return -1;
        }
        std::cout << "Model results resized to " << (resized_outputs[0]).shape() << " for merging" << std::endl;
        merged_output_classes = static_cast<float *>(resized_outputs[0].flat<float>().data());
        break;
      }
      default: LOG(ERROR) << "Error: Invalid number of sub-images: " << sub_images.size();
        return -1;
    }


    // resize model output as percent of actual image dimensions if necessary
    auto final_image_height = static_cast<uint32>(scale_percent * static_cast<double>(image_height) / 100);
    auto final_image_width = static_cast<uint32>(scale_percent * static_cast<double>(image_width) / 100);
//    auto *final_output_classes = new float[final_image_height * final_image_width * output_classes];
    std::vector<Tensor> final_outputs {};
    // resize original image to use for rgb colors (first three channels) in output tiff file
    cv::Mat resized_mat(final_image_width, final_image_height, CV_8UC3);
    if (scale_percent != 100) {
      std::cout << "Scaling not implemented" << std::endl;
      return -1;
//      std::cout << "Model results resized to " << final_image_width << " width x " << final_image_height << " height for output" << std::endl;
////      resize_status = ::hive_segmentation::ResizeTensor(resized_outputs[0], &final_outputs, final_image_height, final_image_width);
////      if (!resize_status.ok()) {
////        LOG(ERROR) << "Error: Resizing final output by scaling factor failed: " << resize_status;
////        return -1;
////      }
//////      auto const &final_output = final_outputs[0];
//////      auto final_output_array = final_outputs[0].flat<float>().data();
//////      final_output_classes = static_cast<float *>(final_output_array);
////      final_output_classes = static_cast<float *>(final_outputs[0].flat<float>().data());
//      std::cout << "Final Model results resized to " << (final_outputs[0]).shape() << " for output" << std::endl;
//      // resize original image for scaled pixel values to use in tiff output
//      cv::resize(orig_image_mat, resized_mat, cv::Size(final_image_width, final_image_height));
//      std::cout << "Merged output Image x width " << resized_mat.cols << std::endl;
//      std::cout << "Merged output Image y height " << resized_mat.rows << std::endl;
//      assert(resized_mat.cols == final_image_width);
//      assert(resized_mat.rows == final_image_height);
    } else {
//      final_output_classes = merged_output_classes;
      resized_mat = orig_image_mat;
    }


    // merge and save data into tiff file
    if (image_result_filename.empty()){
      image_result_filename = image_filename.substr(0, image_filename.find_last_of('.'))+".tif";
      std::cout << "Using input filename " << image_filename << " as basis for output file name: " << image_result_filename << std::endl;
    }
    hive_segmentation::write_tiff_file(merged_output_classes, resized_mat, final_image_height, final_image_width, input_channels, output_classes, image_result_filename);


    // cleanup
    delete[] merged_output_classes;
    merged_output_classes = nullptr;
    delete[] dual_output_classes;
    dual_output_classes = nullptr;
//    delete[] final_output_classes;

  } // end while loop for image filenames

  std::cout << "Done" << std::endl;
  return 0;
}