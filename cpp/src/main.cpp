/*
 * Copyright 2024 AutoML Freiburg
 *
 * Modifications made to this file are listed in https://github.com/automl/CVPR24-MedSAM-on-Laptop/blob/main/cpp/README.md
 *
 * Original code licensed under the Apache License, Version 2.0.
 * See the LICENSE file in this directory for more information.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor-io/xnpz.hpp>

#include "lrucache.hpp"

using namespace std::string_literals;
using ImageSize = std::array<size_t, 2>;

constexpr size_t EMBEDDINGS_CACHE_SIZE = 1024;
constexpr size_t IMAGE_ENCODER_INPUT_SIZE = 256;
const ov::Shape INPUT_SHAPE = {1,3, IMAGE_ENCODER_INPUT_SIZE, IMAGE_ENCODER_INPUT_SIZE};

std::array<size_t, 2> get_preprocess_shape(size_t oldh, size_t oldw) {
  double scale = 1.0 * IMAGE_ENCODER_INPUT_SIZE / std::max(oldh, oldw);
  size_t newh = scale * oldh + 0.5;
  size_t neww = scale * oldw + 0.5;
  return {newh, neww};
}

xt::xtensor<float, 1> get_bbox(xt::xtensor<float, 2>& mask) {
  auto indices = xt::where(mask > 0);
  auto y_indices = indices[0], x_indices = indices[1];
  auto x_min = *std::min_element(x_indices.begin(), x_indices.end());
  auto x_max = *std::max_element(x_indices.begin(), x_indices.end());
  auto y_min = *std::min_element(y_indices.begin(), y_indices.end());
  auto y_max = *std::max_element(y_indices.begin(), y_indices.end());
  return {(float)x_min, (float)y_min, (float)x_max, (float)y_max};
}

xt::xtensor<float, 1> get_bbox2(xt::xtensor<float, 2>& mask, size_t os1, size_t os2) {
  ImageSize new_size = get_preprocess_shape(os1, os2);
  xt::xtensor<uint16_t, 2> bm = xt::cast<uint16_t>(mask > 0.0f);
  cv::Mat mat1(cv::Size(os2, os1), CV_16UC1, bm.data()), mat2;
  cv::resize(mat1, mat2, cv::Size(new_size[1], new_size[0]), 0, 0, cv::INTER_AREA);
  bm=xt::adapt((uint16_t*)mat2.data, mat2.total() * mat2.channels(), xt::no_ownership(), std::vector<int>{mat2.rows, mat2.cols});
  bm= xt::pad(bm, {{0, IMAGE_ENCODER_INPUT_SIZE - new_size[0]}, {0, IMAGE_ENCODER_INPUT_SIZE - new_size[1]}});
  auto indices = xt::where(bm > 0);
  auto y_indices = indices[0], x_indices = indices[1];
  auto x_min = *std::min_element(x_indices.begin(), x_indices.end());
  auto x_max = *std::max_element(x_indices.begin(), x_indices.end());
  auto y_min = *std::min_element(y_indices.begin(), y_indices.end());
  auto y_max = *std::max_element(y_indices.begin(), y_indices.end());
  return {(float)x_min, (float)y_min, (float)x_max, (float)y_max};
}

void apply_bbox_shift(xt::xtensor<float, 1>& box, int shift=3) {
  box[0] = std::max((box[0]-shift), 0.0f);
  box[1] = std::max((box[1]-shift), 0.0f);
  box[2] = std::min((box[2]+shift), (float)IMAGE_ENCODER_INPUT_SIZE);
  box[3] = std::min((box[3]+shift), (float)IMAGE_ENCODER_INPUT_SIZE);
}

template <class T>
T cast_npy_file(xt::detail::npy_file& npy_file) {
  auto m_typestring = npy_file.m_typestring;
  if (m_typestring == "|u1") {
    return npy_file.cast<uint8_t>();
  } else if (m_typestring == "<u2") {
    return npy_file.cast<uint16_t>();
  } else if (m_typestring == "<u4") {
    return npy_file.cast<uint32_t>();
  } else if (m_typestring == "<u8") {
    return npy_file.cast<uint64_t>();
  } else if (m_typestring == "|i1") {
    return npy_file.cast<int8_t>();
  } else if (m_typestring == "<i2") {
    return npy_file.cast<int16_t>();
  } else if (m_typestring == "<i4") {
    return npy_file.cast<int32_t>();
  } else if (m_typestring == "<i8") {
    return npy_file.cast<int64_t>();
  } else if (m_typestring == "<f4") {
    return npy_file.cast<float>();
  } else if (m_typestring == "<f8") {
    return npy_file.cast<double>();
  }
  XTENSOR_THROW(std::runtime_error, "Cast error: unknown format "s + m_typestring);
}

struct Encoder {
  ov::CompiledModel model;
  ov::InferRequest infer_request;
  ImageSize original_size, new_size;

  Encoder(ov::Core& core, const std::string& model_path) {
    model = core.compile_model(model_path, "CPU");
    infer_request = model.create_infer_request();
  }

  void set_sizes(const ImageSize& original_size, const ImageSize& new_size) {
    this->original_size = original_size;
    this->new_size = new_size;
  }

  ov::Tensor encode_image(const ov::Tensor& input_tensor) {
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    return infer_request.get_output_tensor();
  }

  xt::xtensor<float, 3> preprocess_2D(xt::xtensor<uint8_t, 3>& original_img) {
    assert(original_img.shape()[0] == 3);
    cv::Mat mat1(cv::Size(original_size[1], original_size[0]), CV_8UC3, original_img.data()), mat2;
    cv::resize(mat1, mat2, cv::Size(new_size[1], new_size[0]), 0, 0, cv::INTER_AREA);

    xt::xtensor<float, 3> img = xt::adapt((uint8_t*)mat2.data, mat2.total() * mat2.channels(), xt::no_ownership(), std::vector<int>{mat2.rows, mat2.cols, mat2.channels()});
    img = (img - xt::amin(img)()) / std::clamp(xt::amax(img)() - xt::amin(img)(), 1e-8f, 1e18f);
    img = xt::pad(img, {{0, IMAGE_ENCODER_INPUT_SIZE - new_size[0]}, {0, IMAGE_ENCODER_INPUT_SIZE - new_size[1]}, {0, 0}});
    img = xt::transpose(img, {2,0,1});
    return img;
  }

  xt::xtensor<float, 3> preprocess_3D(xt::xtensor<uint8_t, 3>& original_img, int z) {
    auto data = original_img.data() + z * original_size[0] * original_size[1];
    cv::Mat mat1(cv::Size(original_size[1], original_size[0]), CV_8UC1, data), mat2;
    cv::resize(mat1, mat2, cv::Size(new_size[1], new_size[0]), 0, 0, cv::INTER_AREA);

    xt::xtensor<float, 3> img = xt::adapt((uint8_t*)mat2.data, mat2.total(), xt::no_ownership(), std::vector<int>{mat2.rows, mat2.cols, 1});
    img = (img - xt::amin(img)()) / std::clamp(xt::amax(img)() - xt::amin(img)(), 1e-8f, 1e18f);
    // also transpose here
    img = xt::repeat(xt::pad(img, {{0, IMAGE_ENCODER_INPUT_SIZE - new_size[0]}, {0, IMAGE_ENCODER_INPUT_SIZE - new_size[1]}, {0, 0}}), 3, 2);
    img = xt::transpose(img, {2,0,1});
    return img;
  }
};

struct Decoder {
  ov::CompiledModel model;
  ov::CompiledModel model2;
  ov::InferRequest infer_request;
  ov::InferRequest infer_request2;
  ImageSize original_size, new_size;
  xt::xtensor<float, 4> pe;

  Decoder(ov::Core& core, const std::string& model_path, const std::string& model_path2) {
    model = core.compile_model(model_path, "CPU");
    model2 = core.compile_model(model_path2, "CPU");
    infer_request = model.create_infer_request();
    infer_request2 = model2.create_infer_request();
    pe = xt::load_npy<float>("openvinomodels/positional_encoding.npy");
  }

  void set_sizes(const ImageSize& original_size, const ImageSize& new_size) {
    this->original_size = original_size;
    this->new_size = new_size;
  }

  void set_embedding_tensor(const ov::Tensor& embedding_tensor) {
    infer_request2.set_tensor("image_embeddings", embedding_tensor);
  }

  xt::xtensor<float, 2> decode_mask(const ov::Tensor& box_tensor) {
    infer_request.set_tensor("boxes", box_tensor);
    infer_request.infer();
    auto sparse_embeddings = infer_request.get_tensor("sparse_embeddings");
    xt::xtensor<float, 3> se = xt::adapt(sparse_embeddings.data<float>(), 512, xt::no_ownership(), std::vector<int>{1,2,256});
    auto dense_embeddings = infer_request.get_tensor("dense_embeddings");
    infer_request2.set_tensor("sparse_prompt_embeddings", sparse_embeddings);
    infer_request2.set_tensor("dense_prompt_embeddings", dense_embeddings);
    ov::Tensor dense_pe(ov::element::f32, {1,256,64,64}, pe.data());
    infer_request2.set_tensor("image_pe", dense_pe);
    infer_request2.infer();
    

    xt::xtensor<float, 2> mask = xt::adapt(infer_request2.get_tensor("low_res_masks").data<float>(), IMAGE_ENCODER_INPUT_SIZE * IMAGE_ENCODER_INPUT_SIZE, xt::no_ownership(), std::vector<int>{IMAGE_ENCODER_INPUT_SIZE, IMAGE_ENCODER_INPUT_SIZE});
    mask = xt::view(mask, xt::range(_, new_size[0]), xt::range(_, new_size[1]));

    cv::Mat mat1(cv::Size(new_size[1], new_size[0]), CV_32FC1, mask.data()), mat2;
    cv::resize(mat1, mat2, cv::Size(original_size[1], original_size[0]), 0, 0, cv::INTER_LINEAR);
    return xt::adapt((float*)mat2.data, mat2.total(), xt::no_ownership(), std::vector<int>{mat2.rows, mat2.cols});
  }
};

void infer_2d(std::string img_file, std::string seg_file, Encoder& encoder, Decoder& decoder) {
  auto npz_data = xt::load_npz(img_file);
  auto original_img = cast_npy_file<xt::xtensor<uint8_t, 3>>(npz_data["imgs"]);
  auto boxes = cast_npy_file<xt::xtensor<float, 2>>(npz_data["boxes"]);
  assert(boxes.shape()[1] == 4);

  ImageSize original_size = {original_img.shape()[0], original_img.shape()[1]};
  ImageSize new_size = get_preprocess_shape(original_size[0], original_size[1]);
  boxes /= std::max(original_size[0], original_size[1]);
  boxes *= 256;
  boxes = xt::floor(boxes);
  encoder.set_sizes(original_size, new_size);
  decoder.set_sizes(original_size, new_size);

  auto img = encoder.preprocess_2D(original_img);
  ov::Tensor input_tensor(ov::element::f32, INPUT_SHAPE, img.data());
  ov::Tensor embedding_tensor = encoder.encode_image(input_tensor);

  xt::xtensor<uint16_t, 2> segs = xt::zeros<uint16_t>({original_size[0], original_size[1]});
  

  decoder.set_embedding_tensor(embedding_tensor);
  for (int i = 0; i < boxes.shape()[0]; ++i) {
    ov::Tensor box_tensor(ov::element::f32, {1,1,1,4}, boxes.data() + i * 4);
    auto mask = decoder.decode_mask(box_tensor);
    xt::filtration(segs, mask > 0) = i + 1;
  }

  xt::dump_npz(seg_file, "segs", segs, true, false);
}

ov::Tensor copy_ov(ov::Tensor t){
ov::Tensor copied_tensor(t.get_element_type(),
                         t.get_shape());
std::memcpy(copied_tensor.data(),
            t.data(),
            t.get_byte_size());
return copied_tensor;
}

void infer_3d(std::string img_file, std::string seg_file, Encoder& encoder, Decoder& decoder) {
  auto npz_data = xt::load_npz(img_file);
  auto original_img = cast_npy_file<xt::xtensor<uint8_t, 3>>(npz_data["imgs"]);
  auto boxes = cast_npy_file<xt::xtensor<uint16_t, 2>>(npz_data["boxes"]);
  assert(boxes.shape()[1] == 6);

  ImageSize original_size = {original_img.shape()[1], original_img.shape()[2]};
  ImageSize new_size = get_preprocess_shape(original_size[0], original_size[1]);
  encoder.set_sizes(original_size, new_size);
  decoder.set_sizes(original_size, new_size);

  cache::lru_cache<int, ov::Tensor> cached_embeddings(EMBEDDINGS_CACHE_SIZE);
  auto get_embedding = [&](int z) {
    if (!cached_embeddings.exists(z)) {
      auto img = encoder.preprocess_3D(original_img, z);
      ov::Tensor input_tensor(ov::element::f32, INPUT_SHAPE, img.data());
      ov::Tensor embedding_tensor = encoder.encode_image(input_tensor);
      cached_embeddings.put(z, copy_ov(embedding_tensor));
    }
    return copy_ov(cached_embeddings.get(z));
  };
  auto process_slice = [&](int z, xt::xtensor<float, 1>& box) {
    ov::Tensor embedding_tensor = get_embedding(z);
    ov::Tensor box_tensor(ov::element::f32, {1,1,1,4}, box.data());
    decoder.set_embedding_tensor(embedding_tensor);
    return decoder.decode_mask(box_tensor);
  };

  xt::xtensor<uint16_t, 3> segs = xt::zeros_like(original_img);
  for (int i = 0; i < boxes.shape()[0]; ++i) {
    uint16_t x_min = boxes(i, 0), y_min = boxes(i, 1), z_min = boxes(i, 2);
    uint16_t x_max = boxes(i, 3), y_max = boxes(i, 4), z_max = boxes(i, 5);
    z_min = std::max(z_min, uint16_t(0));
    z_max = std::min(z_max, uint16_t(original_img.shape()[0]));
    uint16_t z_middle = (z_min + z_max) / 2;

    xt::xtensor<float, 1> box_default = {(float)x_min, (float)y_min, (float)x_max, (float)y_max};
    box_default /= std::max(original_size[0], original_size[1]);
    box_default *= 256;
    box_default = xt::floor(box_default);
    
    // infer z_middle
    xt::xtensor<float, 1> box_middle;
    {
      auto mask_middle = process_slice(z_middle, box_default);
      xt::filtration(xt::view(segs, z_middle, xt::all(), xt::all()), mask_middle > 0) = i + 1;
      if (xt::amax(mask_middle)() > 0) {
        box_middle = 256*get_bbox(mask_middle) / std::max(original_size[0], original_size[1]);
	// get_bbox2 is equivalent to our old approach, but should be slightly slower and less accurate
        //box_middle = get_bbox2(mask_middle, original_size[0], original_size[1]);
	apply_bbox_shift(box_middle);
      } else {
        box_middle = box_default;
      }
    }

    // infer z_middle+1 to z_max-1
    auto last_box = box_middle;
    for (int z = z_middle+1; z < z_max; ++z) {
      auto mask = process_slice(z, last_box);
      xt::filtration(xt::view(segs, z, xt::all(), xt::all()), mask > 0) = i + 1;
      if (xt::amax(mask)() > 0) {
        last_box = 256*get_bbox(mask) / std::max(original_size[0], original_size[1]);
        //last_box = get_bbox2(mask, original_size[0], original_size[1]);
	apply_bbox_shift(last_box);
      } else {
        last_box = box_default;
      }
    }

    // infer z_min to z_middle-1
    last_box = box_middle;
    for (int z = z_middle - 1; z >= z_min; --z) {
      auto mask = process_slice(z, last_box);
      xt::filtration(xt::view(segs, z, xt::all(), xt::all()), mask > 0) = i + 1;
      if (xt::amax(mask)() > 0) {
        last_box = 256*get_bbox(mask) / std::max(original_size[0], original_size[1]);
        //last_box = get_bbox2(mask, original_size[0], original_size[1]);
	apply_bbox_shift(last_box);
      } else {
        last_box = box_default;
      }
    }
  }

  xt::dump_npz(seg_file, "segs", segs, true, false);
}

bool starts_with(const std::string& str, const std::string& prefix) {
  return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const std::string filename_to_modelname(std::string& filename){
    if (starts_with(filename, "3DBox_PET")) return "3D";
    if (starts_with(filename, "3DBox_MR")) return "3D";
    if (starts_with(filename, "3DBox_CT")) return "3D";

    if (starts_with(filename, "2DBox_X-Ray")) return "XRay";
    if (starts_with(filename, "2DBox_XRay")) return "XRay";
    if (starts_with(filename, "2DBox_CXR")) return "XRay";
    if (starts_with(filename, "2DBox_XR")) return "XRay";
    if (starts_with(filename, "2DBox_US")) return "US";
    if (starts_with(filename, "2DBox_Ultra")) return "US";
    if (starts_with(filename, "2DBox_Fundus")) return "Fundus";
    if (starts_with(filename, "2DBox_Endoscopy")) return "Endoscopy";
    if (starts_with(filename, "2DBox_Endoscope")) return "Endoscopy";

    if (starts_with(filename, "2DBox_Dermoscope")) return "Dermoscopy";
    if (starts_with(filename, "2DBox_Dermoscopy")) return "Dermoscopy";

    if (starts_with(filename, "2DBox_Microscope")) return "Microscopy";
    if (starts_with(filename, "2DBox_Microscopy")) return "Microscopy";

    if (starts_with(filename, "2DBox_CT")) return "3D";
    if (starts_with(filename, "2DBox_MR")) return "3D";
    if (starts_with(filename, "2DBox_PET")) return "3D";

    if (starts_with(filename, "2DBox_Mamm")) return "Mammography";
    if (starts_with(filename, "2DBox_OCT")) return "OCT";

    if (starts_with(filename, "3DBox_")) return "3D";

    if (filename.find("Microscope") != std::string::npos) return "Microscopy";
    if (filename.find("Microscopy") != std::string::npos) return "Microscopy";
    if (filename.find("Dermoscopy") != std::string::npos) return "Dermoscopy";
    if (filename.find("Endoscopy") != std::string::npos) return "Endoscopy";
    if (filename.find("Fundus") != std::string::npos) return "Fundus";
    if (filename.find("X-Ray") != std::string::npos) return "XRay";
    if (filename.find("XRay") != std::string::npos) return "XRay";
    if (filename.find("PET") != std::string::npos) return "3D";
    if (filename.find("OCT") != std::string::npos) return "OCT"; // make sure OCT stays before CT check
    if (filename.find("MR") != std::string::npos) return "3D";
    if (filename.find("Mamm") != std::string::npos) return "Mammography";
    if (filename.find("US") != std::string::npos) return "US";
    if (filename.find("CT") != std::string::npos) return "3D";

    std::cout << filename << " no match found" << std::endl;
    return "general";
}

cache::lru_cache<std::string, Encoder> encoder_cache(16); // 10 should be sufficient, should check lru implementation to make sure
cache::lru_cache<std::string, Decoder> decoder_cache(16);
Encoder load_encoder(std::string model_name, ov::Core core) {
    if (!encoder_cache.exists(model_name)) {
      //std::cout << model_name << std::endl;
      Encoder encoder(core, "openvinomodels/"+model_name+"_image_encoder.xml");
      encoder_cache.put(model_name, encoder);
    }
    return encoder_cache.get(model_name);
};

Decoder load_decoder(std::string model_name, ov::Core core) {
    if (!decoder_cache.exists(model_name)) {
      //std::cout << model_name << std::endl;
      Decoder decoder(core, "openvinomodels/prompt_encoder.xml", "openvinomodels/"+model_name+"_mask_decoder.xml");
      decoder_cache.put(model_name, decoder);
    }
    return decoder_cache.get(model_name);
};

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <model cache folder> <imgs folder> <segs folder>\n";
    return 1;
  }

  ov::Core core;
  core.set_property("CPU", ov::hint::inference_precision(ov::element::f32));
  core.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
  core.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  core.set_property("CPU", ov::hint::num_requests(1));
  core.set_property(ov::cache_dir(argv[1]));

  std::filesystem::path imgs_folder(argv[2]);
  if (!std::filesystem::is_directory(imgs_folder)) {
    throw std::runtime_error(imgs_folder.string() + " is not a folder");
  }

  std::filesystem::path segs_folder(argv[3]);
  if (!std::filesystem::exists(segs_folder) && !std::filesystem::create_directory(segs_folder)) {
    throw std::runtime_error("Failed to create " + segs_folder.string());
  }
  if (!std::filesystem::is_directory(segs_folder)) {
    throw std::runtime_error(segs_folder.string() + " is not a folder");
  }

  for (const auto& entry : std::filesystem::directory_iterator(imgs_folder)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    auto base_name = entry.path().filename().string();
    if (ends_with(base_name, ".npz")) {
      Encoder encoder = load_encoder(filename_to_modelname(base_name), core);
      Decoder decoder = load_decoder(filename_to_modelname(base_name), core);

      auto img_file = entry.path().string();
      auto seg_file = (segs_folder / entry.path().filename()).string();
      if (starts_with(base_name, "2D")) {
        infer_2d(img_file, seg_file, encoder, decoder);
      } else {
        infer_3d(img_file, seg_file, encoder, decoder);
      }
    }
  }

  return 0;
}
