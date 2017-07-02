#include "../../precomp.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <assert.h>
#include "common.hpp"
#include "libdnn.hpp"
#include "benchmark.hpp"
#include "opencl_kernels_dnn.hpp"
#include "greentea_math_functions.hpp"

// #define TEST_ALL_KERNELS

using namespace cv;

namespace greentea
{

#ifdef HAVE_OPENCL
template<typename Dtype>
LibDNNConvSpatial<Dtype>::LibDNNConvSpatial(LibDNNConvConfig config)
{
    bias_term_ = config.bias_term;
    bias_multiplier_ = config.bias_term ? 1.0 : 0.0;
    int_tp dims = config.in_shape.size();
    int_tp spatial_dims = config.kernel.size();

    phase_test_ = config.phase_test;
    num_axes_ = spatial_dims;
    fmaps_in_ = config.in_shape[dims - spatial_dims - 1];
    fmaps_out_ = config.out_shape[dims - spatial_dims - 1];

    group_ = config.group;

    for (int_tp i = 0; i < spatial_dims; ++i) {
        kernel_shape_.push_back(config.kernel[i]);
        pad_.push_back(config.pad[i]);
        stride_.push_back(config.stride[i]);
        dilation_.push_back(config.dilation[i]);
        im_in_shape_.push_back(config.in_shape[dims - spatial_dims + i]);
        im_out_shape_.push_back(config.out_shape[dims - spatial_dims + i]);
    }

    bestKernelConfig = NULL;
    bias_ = NULL;
    tuned_ = false;
    try_cache_ = false;
    swizzled_weights_ = NULL;
    channels_ = fmaps_in_;
    num_output_ = fmaps_out_;
    kernel_dim_ = fmaps_in_ / group_;
    in_spatial_dim_ = 1;
    out_spatial_dim_ = 1;
    for (int_tp i = 0; i < spatial_dims; ++i) {
        kernel_dim_ *= config.kernel[i];
        in_spatial_dim_ *= config.in_shape[dims - spatial_dims + i];
        out_spatial_dim_ *= config.out_shape[dims - spatial_dims + i];
    }

    // assumption: spatial dimension is 2.
    kernel_h_ = kernel_shape_[0];
    kernel_w_ = kernel_shape_[1];
    pad_h_ = pad_[0];
    pad_w_ = pad_[1];
    stride_h_ = stride_[0];
    stride_w_ = stride_[1];
    dilation_h_ = dilation_[0];
    dilation_w_ = dilation_[1];

    M_ = num_output_ / group_;
    K_ = channels_ * kernel_h_ * kernel_w_ / group_;

    height_ = im_in_shape_[0];
    width_ = im_in_shape_[1];
    const int_tp kernel_extent_h = dilation_h_ * (kernel_h_ - 1) + 1;
    const int_tp kernel_extent_w = dilation_w_ * (kernel_w_ - 1) + 1;
    output_h_ = (height_ + 2 * pad_h_ - kernel_extent_h) / stride_h_ + 1;
    output_w_ = (width_ + 2 * pad_w_ - kernel_extent_w) / stride_w_ + 1;

    bottom_dim_ = channels_ * in_spatial_dim_;
    top_dim_ = num_output_ * out_spatial_dim_;

    if (std::getenv("CLCAFFE_CACHE_PATH"))
        cache_path_ << std::getenv("CLCAFFE_CACHE_PATH");
    else if (std::getenv("VIENNACL_CACHE_PATH"))
        cache_path_ << std::getenv("VIENNACL_CACHE_PATH") << "/clCaffe";
    else if (std::getenv("HOME")) {
        cache_path_ << std::getenv("HOME") << "/.cache/clCaffe";
    }
    cache_path_ << "/spatialkernels/";
    struct stat stat_buf;
    bool hasCacheDir = false;
    if (0 != stat(cache_path_.str().c_str(), &stat_buf)) {
        hasCacheDir = !mkdir(cache_path_.str().c_str(), 0755);
    } else if (S_ISDIR(stat_buf.st_mode)) {
        hasCacheDir = true;
    }

    if (hasCacheDir != true) {
        std::cout << "Failed to create cache directory,"
            << "will tune again for next running" << std::endl;
        return;
    }
}

template<typename Dtype>
LibDNNConvSpatial<Dtype>::~LibDNNConvSpatial()
{
    if (swizzled_weights_) {
        clReleaseMemObject((cl_mem)swizzled_weights_);
    }
    if (bestKernelConfig) {
        delete bestKernelConfig;
    }
}

template<typename Dtype>
std::string LibDNNConvSpatial<Dtype>::generate_header()
{
    std::stringstream ss;

    if (std::is_same<Dtype, double>::value) {
        // Test/enable KHR 64 bit (double)
        ss << "#if defined(cl_khr_fp64)" << std::endl;
        ss << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl;
        ss << "#define DOUBLE_SUPPORT_AVAILABLE" << std::endl;

        // Test/enable AMD 64 bit (double)
        ss << "#elif defined(cl_amd_fp64)" << std::endl;
        ss << "#pragma OPENCL EXTENSION cl_amd_fp64 : enable" << std::endl;
        ss << "#define DOUBLE_SUPPORT_AVAILABLE" << std::endl;
        ss << "#endif" << std::endl;
    }

    // Test/enable 32 bit atomics
    ss << "#if defined(cl_khr_int32_base_atomics)" << std::endl;
    ss << "#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable"
       << std::endl;
    ss << "#define ATOMICS_32_AVAILABLE" << std::endl;
    ss << "#endif" << std::endl;
    ss << "#if defined(cl_khr_global_int32_base_atomics)" << std::endl;
    ss << "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable"
       << std::endl;
    ss << "#define ATOMICS_32_AVAILABLE" << std::endl;
    ss << "#endif" << std::endl;

    // 64 bit integers
    if (sizeof(int_tp) == 8 || std::is_same<Dtype, double>::value) {
        // Test/enable 64 bit atomics
        ss << "#if defined(cl_khr_int64_base_atomics)" << std::endl;
        ss << "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable"
           << std::endl;
        ss << "#define ATOMICS_64_AVAILABLE" << std::endl;
        ss << "#endif" << std::endl;
    }

    if (std::is_same<Dtype, double>::value) {
        ss << "#define Dtype double" << std::endl;
        ss << "#define Dtype1 double" << std::endl;
        // double2, double4, double8, double16
        for (int_tp i = 2; i <= 16; i *= 2) {
            ss << "#define Dtype" << i << " double" << i << std::endl;
        }
    } else {
        ss << "#define Dtype float" << std::endl;
        ss << "#define Dtype1 float" << std::endl;
        // float2, float4, float8, float16
        for (int_tp i = 2; i <= 16; i *= 2) {
            ss << "#define Dtype" << i << " float" << i << std::endl;
        }
    }

    if (sizeof(int_tp) == 8) {
        ss << "#define int_tp long" << std::endl;
        ss << "#define uint_tp unsigned long" << std::endl;
        ss << "#define int_tpc long" << std::endl;
        ss << "#define uint_tpc unsigned long" << std::endl;
    } else {
        ss << "#define int_tp int" << std::endl;
        ss << "#define uint_tp unsigned int" << std::endl;
        ss << "#define int_tpc int" << std::endl;
        ss << "#define uint_tpc unsigned int" << std::endl;
    }

    return ss.str();
}

template<typename Dtype>
std::string LibDNNConvSpatial<Dtype>::generate_fw_defs()
{
    std::stringstream ss;

    ss << "#define __CAT(x, y) x##y" << std::endl;
    ss << "#define CAT(x, y) __CAT(x, y)" << std::endl;
    ss << "#define LOOP0(VAR, STMT)" << std::endl;
    ss << "#define LOOP1(VAR, STMT) (STMT); (VAR)++;" << std::endl;
    ss << "#define LOOP2(VAR, STMT) LOOP1(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP3(VAR, STMT) LOOP2(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP4(VAR, STMT) LOOP3(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP5(VAR, STMT) LOOP4(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP6(VAR, STMT) LOOP5(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP7(VAR, STMT) LOOP6(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP8(VAR, STMT) LOOP7(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP9(VAR, STMT) LOOP8(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP10(VAR, STMT) LOOP9(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP11(VAR, STMT) LOOP10(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP12(VAR, STMT) LOOP11(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP13(VAR, STMT) LOOP12(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP14(VAR, STMT) LOOP13(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP15(VAR, STMT) LOOP14(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP16(VAR, STMT) LOOP15(VAR, STMT); (STMT); (VAR)++;"
       << std::endl;
    ss << "#define LOOP(N, VAR, STMT) CAT(LOOP, N)((VAR), (STMT))"
       << std::endl;

    add_def(ss, "KERNEL_WIDTH", kernel_w_);
    add_def(ss, "KERNEL_HEIGHT" , kernel_h_);
    add_def(ss, "STRIDE_X", stride_w_);
    add_def(ss, "STRIDE_Y", stride_h_);
    add_def(ss, "DILATION_X", dilation_w_);
    add_def(ss, "DILATION_Y", dilation_h_);
    add_def(ss, "INPUT_PAD_W", pad_w_);
    add_def(ss, "INPUT_PAD_H", pad_h_);

    return ss.str();
}

typedef enum {
    KERNEL_TYPE_INTEL_IDLF = 2,
    KERNEL_TYPE_BASIC = 4,
    KERNEL_TYPE_GEMM_LIKE = 5
} libdnnConvSpatialKernelType_t;

template<typename Dtype>
std::string LibDNNConvSpatial<Dtype>::generate_fw_kernels(int_tp kernelType,
                                                          int_tp blockM,
                                                          int_tp blockK,
                                                          int_tp blockN)
{
    std::stringstream ss;
    std::stringstream opts;
    std::string kernelUKey;
    int_tp simd_size;

    if (kernelType == KERNEL_TYPE_INTEL_IDLF) {
        simd_size = blockN;
        kernelUKey = generate_specific_key(2, blockM, blockK, 1);

        // kernel name
        kernel_name_ = "IDLF_";
        kernel_name_ += kernelUKey.c_str();
        if (simd_size == 16)
            kernel_name_ += "_SIMD16";
        else
            kernel_name_ += "_SIMD8";

        // options
        opts << "-cl-fast-relaxed-math -D convolve_simd=" << kernel_name_;
        if (IsBeignet())
            opts << " -D__BEIGNET__ ";
        else
            opts << " -cl-no-subgroup-ifp ";
        options_ = opts.str();

        // defs
        int_tp output_width = output_w_;
        int_tp output_height = output_h_;
        int_tp output_block_width = blockM;
        int_tp output_block_height = blockK;
        const int_tp last_block_width = (output_width % output_block_width == 0) ?
                                        output_block_width : output_width % output_block_width;
        const int_tp last_block_height = (output_height % output_block_height == 0) ?
                                         output_block_height : output_height % output_block_height;
        int tile_x = (((output_block_width - 1) * stride_w_ + kernel_w_ * dilation_w_) + 3) & ~3;
        int tile_y = (output_block_height -1) * stride_h_ + kernel_h_ * dilation_h_;
        int tile_y_stride = (4 * simd_size) / tile_x;
        int invec_size = (tile_y + tile_y_stride - 1) / tile_y_stride;

        add_def(ss, "SIMD_SIZE", simd_size);
        add_def(ss, "filter_qualifier", "__global");
        add_def(ss, "OUT_BLOCK_WIDTH", output_block_width);
        add_def(ss, "OUT_BLOCK_HEIGHT", output_block_height);
        add_def(ss, "LAST_BLOCK_WIDTH", last_block_width);
        add_def(ss, "LAST_BLOCK_HEIGHT", last_block_height);
        add_def(ss, "INPUT_DEPTH", channels_ / group_);
        add_def(ss, "TOTAL_INPUT_DEPTH_SIZE", channels_);
        add_def(ss, "TOTAL_OUTPUT_DEPTH", num_output_);
        add_def(ss, "INPUT_START_X", 0);
        add_def(ss, "INPUT_START_Y", 0);
        add_def(ss, "INPUT_START_Z", 0);
        add_def(ss, "NUM_FILTERS", M_);
        add_def(ss, "OUT_BUFF_OFFSET", 0);
        add_def(ss, "TILE_X", tile_x);
        add_def(ss, "TILE_Y", tile_y);
        add_def(ss, "TILE_Y_STRIDE", tile_y_stride);
        add_def(ss, "INVEC_SIZE", invec_size);
        add_def(ss, "ALIGNED_NUM_FILTERS", ALIGN(M_, simd_size));
        add_def(ss, "OUT_BLOCK_SIZE", (output_block_width*output_block_height));

        // kernel source
        // Each work-item computes
        // a OUT_BLOCK_WIDTH * OUT_BLOCK_HEIGHT region of one output map.
        // Each work-group (which will be mapped to 1 SIMD16/SIMD8 EU thread)
        // will compute 16/8 different feature maps,
        // but each feature map is for the same region of the imput image.
        // NDRange:  (output_width+pad)/ OUT_BLOCK_WIDTH,
        //           (output_height+pad)/OUT_BLOCK_HEIGHT,
        //           NUM_FILTERS/OUT_BLOCK_DEPTH
        // NOTE: for beignet
        // this reqd_work_group_size does not guarantee that
        // SIMD16/8 mode will be used,
        // the compiler could choose to use two SIMD8 threads,
        // and if that happens the code will break.
        ss << "#define activation_function(x) (x)" << std::endl;
        ss << "__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))" << std::endl;
        ss << "kernel void" << std::endl;
        ss << "convolve_simd(" << std::endl;
        ss << "__global float* inputs_base," << std::endl;
        ss << "filter_qualifier float* weights_base," << std::endl;
        ss << "__global float* biases_base," << std::endl;
        ss << "__global float* outputs_base," << std::endl;
        ss << "const ushort input_width," << std::endl;
        ss << "const ushort input_height," << std::endl;
        ss << "const ushort output_width," << std::endl;
        ss << "const ushort output_height)" << std::endl;
        ss << "{" << std::endl;
        ss << "__global float* outputs = outputs_base;" << std::endl;
        ss << "__global float* inputs = inputs_base;" << std::endl;
        ss << "filter_qualifier float* weights = weights_base;" << std::endl;
        ss << "__global float* biases = biases_base;" << std::endl;
        // oc = Output Column
        ss << "uint_tp oc = get_global_id(0) * OUT_BLOCK_WIDTH;" << std::endl;
        // or = Output Row
        ss << "uint_tp or = get_global_id(1) * OUT_BLOCK_HEIGHT;" << std::endl;
        // fm = Feature Map = od = Output Depth
        ss << "uint_tp fm = get_global_id(2);" << std::endl;
        ss << "uint_tp fmg = get_group_id(2);" << std::endl;
        ss << "uint_tp lid = get_local_id(2);" << std::endl;
        ss << "float out[OUT_BLOCK_SIZE];" << std::endl;
        ss << "int_tp in_addr;" << std::endl;
        // find weights adress of given neuron (lid is index)
        ss << "uint_tp weight_addr = (fmg % (ALIGNED_NUM_FILTERS/SIMD_SIZE)) * "
           << "INPUT_DEPTH * KERNEL_WIDTH * KERNEL_HEIGHT * SIMD_SIZE + lid;"
           << std::endl;
        ss << "for(int_tp i=0;i<OUT_BLOCK_SIZE;i++) {" << std::endl;
        ss << "out[i]=0.0f;" << std::endl;
        ss << "}" << std::endl;
        ss << "uint_tp num_in_batch = ( fm ) / ALIGNED_NUM_FILTERS;" << std::endl;
        ss << "uint_tp input_batch_offset = "
           << "num_in_batch * input_height * input_width * TOTAL_INPUT_DEPTH_SIZE;"
           << std::endl;
        ss << "int curr_y = or * STRIDE_Y + INPUT_START_Y + (lid / (TILE_X/4));"
           << std::endl;
        ss << "int curr_x = oc * STRIDE_X + INPUT_START_X + (lid % (TILE_X/4)) * 4;"
           << std::endl;
        ss << "#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0" << std::endl;
        ss << "int saved_y = curr_y;" << std::endl;
        ss << "#endif" << std::endl;
        ss << "in_addr = "
           << "input_batch_offset + INPUT_START_Z * input_height * input_width"
            // y tile offset
           << "+  (curr_y - INPUT_PAD_H) * input_width"
            // x tile offset
           << "+   curr_x - INPUT_PAD_W;"
           << std::endl;
        ss << "union {" << std::endl;
        ss << "float4 in_vec[INVEC_SIZE];" << std::endl;
        ss << "float in_array[INVEC_SIZE * 4];" << std::endl;
        ss << "} in_buf;" << std::endl;
        ss << "for(int_tp kd = 0; kd < INPUT_DEPTH; kd++)" << std::endl;
        ss << "{" << std::endl;
        ss << "int_tp in_offset = in_addr;" << std::endl;
        ss << "int_tp reg = 0;" << std::endl;
        ss << "LOOP(INVEC_SIZE, reg," << std::endl;
        ss << "{" << std::endl;
        ss << "#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0" << std::endl;
        ss << "if (curr_y >= INPUT_PAD_H && "
           << "curr_y < input_height + INPUT_PAD_H && "
           << "curr_x + 3 >= INPUT_PAD_W && "
           << "curr_x < input_width + INPUT_PAD_W) {" << std::endl;
        ss << "if (curr_x < INPUT_PAD_W) {" << std::endl;
        ss << "in_buf.in_vec[reg].s0 = 0;" << std::endl;
        ss << "if (curr_x + 1 >= INPUT_PAD_W)" << std::endl;
        ss << "in_buf.in_vec[reg].s1 = *(inputs + in_offset + 1);" << std::endl;
        ss << "else" << std::endl;
        ss << "in_buf.in_vec[reg].s1 = 0;" << std::endl;
        ss << "if (curr_x + 2 >= INPUT_PAD_W)" << std::endl;
        ss << "in_buf.in_vec[reg].s2 = *(inputs + in_offset + 2);" << std::endl;
        ss << "else" << std::endl;
        ss << "in_buf.in_vec[reg].s2 = 0;" << std::endl;
        ss << "in_buf.in_vec[reg].s3 = *(inputs + in_offset + 3);" << std::endl;
        ss << "} else {" << std::endl;
        // read SIMD_SIZE elements
        ss << "in_buf.in_vec[reg] = *(global float4*)(inputs + in_offset);"
           << std::endl;
        ss << "if (curr_x + 1 >= input_width + INPUT_PAD_W)" << std::endl;
        ss << "in_buf.in_vec[reg].s1 = 0;" << std::endl;
        ss << "if (curr_x + 2 >= input_width + INPUT_PAD_W)" << std::endl;
        ss << "in_buf.in_vec[reg].s2 = 0;" << std::endl;
        ss << "if (curr_x + 3 >= input_width + INPUT_PAD_W)" << std::endl;
        ss << "in_buf.in_vec[reg].s3 = 0;" << std::endl;
        ss << "}" << std::endl;
        ss << "} else {" << std::endl;
        ss << "in_buf.in_vec[reg] = 0;" << std::endl;
        ss << "}" << std::endl;
        ss << "curr_y += TILE_Y_STRIDE;" << std::endl;
        ss << "#else" << std::endl;
        // read SIMD_SIZE elements
        ss << "in_buf.in_vec[reg] = *(global float4*)(inputs + in_offset);"
           << std::endl;
        ss << "#endif" << std::endl;
        ss << "in_offset += input_width * TILE_Y_STRIDE;" << std::endl;
        ss << "});" << std::endl;
        ss << "in_addr += input_height * input_width;" << std::endl;
        ss << "#if INPUT_PAD_W != 0 || INPUT_PAD_H != 0" << std::endl;
        ss << "curr_y = saved_y;" << std::endl;
        ss << "#endif" << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT != 1" << std::endl;
        ss << "#define WEIGHT_PREF 8" << std::endl;
        ss << "#else" << std::endl;
        ss << "#define WEIGHT_PREF 1" << std::endl;
        ss << "#endif" << std::endl;
        ss << "union {" << std::endl;
        ss << "float w[WEIGHT_PREF];" << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT != 1" << std::endl;
        ss << "uint8 ui8;" << std::endl;
        ss << "#endif" << std::endl;
        ss << "} weight_buf;" << std::endl;
        ss << "int_tp w_idx=0;" << std::endl;
        ss << "uint_tp orig_weight_addr = weight_addr;" << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT != 1" << std::endl;
        ss << "weight_buf.ui8 = "
           << "intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);"
           << std::endl;
        ss << "weight_addr += SIMD_SIZE * WEIGHT_PREF;" << std::endl;
        ss << "#else" << std::endl;
        ss << "weight_buf.w[0] = as_float("
           << "intel_sub_group_block_read((__global uint *)&weights[weight_addr]));"
           << std::endl;
        ss << "weight_addr += SIMD_SIZE * 1;" << std::endl;
        ss << "#endif" << std::endl;
        ss << "#define BLOCK_IN(n) "
           << "sub_group_broadcast("
           << "in_buf.in_array[((n)%4) + ((n) / (TILE_Y_STRIDE * TILE_X)) * 4], "
           << "(((n) % (TILE_Y_STRIDE * TILE_X))/4))" << std::endl;
        // kr = Kernel Row
        ss << "int_tp kr = 0;" << std::endl;
        ss << "LOOP(KERNEL_HEIGHT, kr," << std::endl;
        ss << "{" << std::endl;
        // kc = Kernel Column
        ss << "int_tp kc = 0;" << std::endl;
        ss << "LOOP(KERNEL_WIDTH, kc," << std::endl;
        ss << "{" << std::endl;
        ss << "for(int_tp br=0; br < OUT_BLOCK_HEIGHT; br++) {" << std::endl;
        ss << "for(int_tp bc=0; bc < OUT_BLOCK_WIDTH; bc++) {" << std::endl;
        ss << "float input = BLOCK_IN((br * STRIDE_Y + kr * DILATION_Y) * "
           << "TILE_X + bc * STRIDE_X + kc * DILATION_X);" << std::endl;
        ss << "out[br * OUT_BLOCK_WIDTH + bc] = "
           << "mad(weight_buf.w[w_idx % WEIGHT_PREF], "
           << "input, out[br * OUT_BLOCK_WIDTH + bc]);" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT > WEIGHT_PREF" << std::endl;
        // We assume KERNEL_WIDTH is equal to KERNEL_HEIGHT here.
        ss << "if ((w_idx + 1) % WEIGHT_PREF == 0" << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT % 8 != 0" << std::endl;
        ss << "&& ((w_idx + 1) <= (KERNEL_WIDTH * KERNEL_HEIGHT - WEIGHT_PREF))"
           << std::endl;
        ss << "#endif" << std::endl;
        ss << ") {" << std::endl;
        ss << "weight_buf.ui8 = "
           << "intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);"
           << std::endl;
        // weights must be stored in just the right SIMD swizzled format
        // for this to work, see host code for details.
        ss << "weight_addr += SIMD_SIZE * WEIGHT_PREF;" << std::endl;
        ss << "}" << std::endl;
        ss << "#if KERNEL_WIDTH*KERNEL_HEIGHT % 8 == 0" << std::endl;
        // need to do nothing
        ss << "#else" << std::endl;
        ss << "else if ((w_idx + 1) %  WEIGHT_PREF == 0 && "
           << "((w_idx + 1) > (KERNEL_WIDTH * KERNEL_HEIGHT - WEIGHT_PREF)))"
           << std::endl;
        ss << "#if KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 1" << std::endl;
        ss << "weight_buf.w[0] = weights[weight_addr];" << std::endl;
        ss << "#elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 == 2" << std::endl;
        ss << "weight_buf.ui8.s01 = "
           << "intel_sub_group_block_read2((__global uint *)&weights[weight_addr]);"
           << std::endl;
        ss << "#elif KERNEL_WIDTH * KERNEL_HEIGHT % 8 <= 4" << std::endl;
        ss << "weight_buf.ui8.s0123 = "
           << "intel_sub_group_block_read4((__global uint *)&weights[weight_addr]);"
           << std::endl;
        ss << "#else" << std::endl;
        ss << "weight_buf.ui8 = "
           << "intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);"
           << std::endl;
        ss << "#endif" << std::endl;
        ss << "#endif" << std::endl;
        ss << "#endif" << std::endl;
        ss << "++w_idx;" << std::endl;
        ss << "});" << std::endl;
        ss << "});" << std::endl;
        ss << "weight_addr = "
           << "orig_weight_addr + KERNEL_WIDTH * KERNEL_HEIGHT * SIMD_SIZE;"
           << std::endl;
        ss << "}" << std::endl;
        // dead code to work around possible compiler bug.
        ss << "if (ALIGNED_NUM_FILTERS != NUM_FILTERS && fm > 0xfffffffeul) {"
           << std::endl;
        ss << "outputs[0] = BLOCK_IN(fm % SIMD_SIZE);" << std::endl;
        ss << "}" << std::endl;
        ss << "fm = fm % ALIGNED_NUM_FILTERS;" << std::endl;
        ss << "if ((ALIGNED_NUM_FILTERS == NUM_FILTERS || fm < NUM_FILTERS)) {"
           << std::endl;
        ss << "uint_tp out_addr = "
           << "OUT_BUFF_OFFSET + "
           << "( num_in_batch * TOTAL_OUTPUT_DEPTH + fm ) * "
           << "output_width * output_height;"
           << std::endl;
        ss << "out_addr += or * output_width + oc;" << std::endl;
        ss << "float bias = biases[fm];" << std::endl;
        ss << "for(uint_tp r = 0; r < OUT_BLOCK_HEIGHT; r++) {" << std::endl;
        ss << "if (r + or >= output_height) break;" << std::endl; 
        ss << "for(uint_tp c = 0; c < OUT_BLOCK_WIDTH; c++) {" << std::endl;
        ss << "if (c + oc >= output_width) break;" << std::endl;
        ss << "outputs[out_addr + r * output_width + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
    } else if (kernelType == KERNEL_TYPE_GEMM_LIKE) {
        simd_size = blockK;
        kernelUKey = generate_specific_key(kernelType, blockM, blockK, blockN);
        // kernel name
        kernel_name_ = "U_GEMM_LIKE_CONV_";
        kernel_name_ += kernelUKey.c_str();
        if (simd_size == 8)
            kernel_name_ += "_SIMD8";
        else
            kernel_name_ += "_SIMD16";

        // kernel specific options
        std::stringstream kernelDef;
        kernelDef << "GEMM_LIKE_CONV_" << blockN << "_" << blockM;
        if (simd_size == 8) {
            kernelDef << "_SIMD8";
        } else {
            kernelDef << "_SIMD16";
        }
        opts << "-cl-fast-relaxed-math -cl-mad-enable -D "
            << kernelDef.str() << " -D Conv_Interleaved="
            << kernel_name_.c_str();
        options_ = opts.str();

        int_tp tile_n_last_div8 = (M_ % 32) / 8;
        add_def(ss, "INPUT_DEPTH", channels_);
        add_def(ss, "WIDTH1", M_);
        add_def(ss, "OUT_PADDING_LEFT", 0);
        add_def(ss, "OUT_PADDING_HEIGHT", 0);
        add_def(ss, "OUT_DEPTH", M_);
        add_def(ss, "KERNEL_WIDTH_DIV2", kernel_w_ / 2);
        add_def(ss, "KERNEL_SLICE_DIV2", (kernel_w_*kernel_h_)/2);
        add_def(ss, "TILE_N_LAST", M_ % 32);
        add_def(ss, "TILE_N_LAST_DIV8", tile_n_last_div8);
        add_def(ss, "TILE_M", blockM);
        add_def(ss, "TILE_N_PER_LANE", 32 / simd_size);

        #define TYPEDEF_FLOAT_N(ele_num) \
        do { \
            ss << "typedef struct float" << ele_num << " { "; \
                for (int i = 0; i < ele_num; i++) { ss << "float s" << i << "; ";} \
                    ss << "} float" << ele_num << ";" << std::endl; \
        } while (0)

        TYPEDEF_FLOAT_N(1);
        TYPEDEF_FLOAT_N(5);
        TYPEDEF_FLOAT_N(6);
        TYPEDEF_FLOAT_N(7);
        TYPEDEF_FLOAT_N(9);
        TYPEDEF_FLOAT_N(10);
        TYPEDEF_FLOAT_N(11);
        TYPEDEF_FLOAT_N(12);
        TYPEDEF_FLOAT_N(13);
        TYPEDEF_FLOAT_N(14);
        TYPEDEF_FLOAT_N(15);
        // never used but makes compiler happy.
        ss << "typedef struct float0 { float s0; } float0;" << std::endl;

        add_def(ss, "OUT_PITCH_X", "output_width");
        add_def(ss, "OUT_PITCH_Y", "(output_width * output_height)");
        add_def(ss, "ROW_PITCH", "input_width");
        add_def(ss, "SLICE_PITCH", "(input_width * input_height)");
        add_def(ss, "TILE_K", kernel_w_);
        add_def(ss, "TILE_N", 32);
        add_def(ss, "OUT_PITCH_Z",
                "(output_width * output_height * OUT_DEPTH)");
        add_def(ss, "ALIGNED_INPUT_SIZE",
                "(input_height * input_width * INPUT_DEPTH)");

        std::vector<std::string> elems16({
                "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
                "s8", "s9", "sa", "sb", "sc", "sd", "se", "sf" });

        #define GENERATE_DOT_PRODUCT(ele_num) \
        do { \
            ss << "#define DOT_PRODUCT_" << ele_num \
                << "( _result, _rowA, colB ) {   "; \
                for (int i = 0; i < ele_num; i++) { \
                    if (i < 10) {\
                        ss << "_result.s" << i \
                            << " = mad( _rowA, sub_group_broadcast( colB, " << i \
                            << "), _result.s" << i << " );"; \
                    } else {\
                        ss << "_result." << elems16[i] \
                            << " = mad( _rowA, sub_group_broadcast( colB, " << i \
                            << "), _result." << elems16[i] << " );"; \
                    }\
                } \
            ss << "    }" << std::endl; \
        } while (0)

        GENERATE_DOT_PRODUCT(8);
        GENERATE_DOT_PRODUCT(16);

        // kernel source
        if (simd_size == 8)
            ss << "__attribute__((intel_reqd_sub_group_size(8)))" << std::endl;
        else if (!IsBeignet())
            ss << "__attribute__((intel_reqd_sub_group_size(16)))" << std::endl;
        ss << "__kernel void Conv_Interleaved(" << std::endl;
        ss << "const __global float *src0," << std::endl;
        ss << "const __global float *src1," << std::endl;
        ss << "const __global float *biases," << std::endl;
        ss << "__global float *dst," << std::endl;
        ss << "const ushort input_width," << std::endl;
        ss << "const ushort input_height," << std::endl;
        ss << "const ushort output_width," << std::endl;
        ss << "const ushort output_height)" << std::endl;
        ss << "{" << std::endl;
        ss << "const int group_x = get_group_id(0);" << std::endl;
        ss << "const int group_y = get_group_id(1);" << std::endl;
        ss << "const int global_x = get_global_id(0);" << std::endl;
        ss << "const int global_y = get_global_id(1);" << std::endl;
        ss << "const int global_z = get_global_id(2);" << std::endl;
        ss << "int interleaved_y;" << std::endl;
        ss << "int kernel_y;" << std::endl;
        ss << "int kernel_idx;" << std::endl;
        ss << "typedef CAT( float, KERNEL_WIDTH ) float_t;" << std::endl;
        // True for all threads if filter_width is multiple of TILE_N
        // else, true for all but right-most column of threads.
        ss << "if( TILE_N_LAST == 0 || global_x < WIDTH1 / TILE_N ) " << std::endl;
        ss << "{" << std::endl;
        // Result ctile (*dst) is M rows x N columns
        // LWG size is 1x8 or 1x16.
        // Thus each thread calculates (8 or 16) *M rows x N cols of ctile.
        if (simd_size == 16) {
            ss << "float16  blockC00 = 0.f;" << std::endl;
            ss << "float16  blockC10 = 0.f;" << std::endl;
        } else {
            ss << "float8  blockC00 = 0.f;" << std::endl;
            ss << "float8  blockC10 = 0.f;" << std::endl;
            ss << "float8  blockC20 = 0.f;" << std::endl;
            ss << "float8  blockC30 = 0.f;" << std::endl;
        }
        if (blockM == 2 && simd_size == 8) {
            ss << "float8  blockC01 = 0.f;" << std::endl;
            ss << "float8  blockC11 = 0.f;" << std::endl;
            ss << "float8  blockC21 = 0.f;" << std::endl;
            ss << "float8  blockC31 = 0.f;" << std::endl;
        }
        // Src0 (patch input) is directly used as atile.
        // Each work item points to the start of a different patch.
        // atile is M rows x K columns." << std::endl
        ss << "int curr_x = ( (global_y * TILE_M) % output_width ) * STRIDE_X;"
           << std::endl;
        ss << "int curr_y = ( (global_y * TILE_M) / output_width ) * STRIDE_Y;"
           << std::endl;
        if (blockM == 2) {
            ss << "int curr_x1 = ((global_y * TILE_M + 1) % output_width) * STRIDE_X;"
               << std::endl;
            ss << "int curr_y1 = ((global_y * TILE_M + 1) / output_width) * STRIDE_Y;"
               << std::endl;
        }
        if (pad_h_ != 0 || pad_w_ != 0 || dilation_w_ != 1 || dilation_h_ != 1) {
            ss << "int saved_y = curr_y;" << std::endl;
            if (blockM == 2) {
                ss << "int saved_y1 = curr_y1;" << std::endl;
            }
        }
        ss << "const __global float *src0_read = src0" << std::endl;
        // batch offset
        ss << "+ ALIGNED_INPUT_SIZE * global_z" << std::endl;
        // y offset
        ss << "+ (curr_y - INPUT_PAD_H) * ROW_PITCH" << std::endl;
        // x offset
        ss << "+ (curr_x - INPUT_PAD_W);" << std::endl;
        if (blockM == 2) {
            ss << "const __global float *src0_read1 = src0" << std::endl;
            // batch offset
            ss << "+ ALIGNED_INPUT_SIZE * global_z" << std::endl;
            // y offset
            ss << "+ (curr_y1 - INPUT_PAD_H) * ROW_PITCH" << std::endl;
            // x offset
            ss << "+ curr_x1 - INPUT_PAD_W;" << std::endl;
        }
        // Src1 (filter) is directly used as btile.
        // It starts at the top of src1 and walks down.
        // btile is K rows x N columns.
        ss << "const __global float *src1_read = src1 + ( global_x * TILE_N  * 2);"
            << std::endl;
        // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
        // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch
        // and KERNEL_WIDTH/2 rows of interleaved filter.
        ss << "int patch_depth = 0;" << std::endl;
        if (!IsBeignet() && simd_size == 16)
            ss << "__attribute__((opencl_unroll_hint(1)))" << std::endl;
        ss << "do" << std::endl;
        ss << "{" << std::endl;
        ss << "int patch_row = 0;" << std::endl;
        if (pad_h_ != 0 || pad_w_ != 0 || dilation_w_ != 1 || dilation_h_ != 1) {
            ss << "curr_y = saved_y;" << std::endl;
            if (blockM == 2)
                ss << "curr_y1 = saved_y1;" << std::endl;
        }
        if (!IsBeignet() && simd_size == 16)
            ss << "__attribute__((opencl_unroll_hint(1)))" << std::endl;
        ss << "do" << std::endl;
        ss << "{" << std::endl;
        /*
         * Load atile and btile.
         *
         * Kernel data is partially interleaved. 
         * Every 2 rows are interleaved at float8 granularity.
         * The exception is that if KERNEL_WIDTH is odd the last row is not
         * interleaved.
         * The non interleaved row is padded with zero to ensure same size
         * as interleaved rows.
         * This interleaving is done to ensure 0% GDR bank conflicts.
         * For example, this is how the
         * kernel data would be arranged before/after interleaving for
         * KERNEL_WIDTH=3.
         * (0, 0) (8, 0) (16, 0) (24, 0) ...    (0, 0) (0, 1) (8, 0) (8, 1)
         * (0, 1) (8, 1) (16, 1) (24, 1) ... => (0, 2) (8, 2) (16, 2) (24, 2) ...
         * (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
         * ...
         */
        ss << "const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;"
           << std::endl;
        if (pad_h_ == 0 && pad_w_ == 0 && dilation_w_ == 1 && dilation_h_ == 1) {
            ss << "float_t blockA00 = ( (const __global float_t*)src0_read )[0];"
               << std::endl;
            ss << "float*  pblockA00 = (float*)(&blockA00);" << std::endl;
            if (blockM == 2) {
                ss << "float_t blockA01 = ( (const __global float_t*)src0_read1 )[0];"
                   << std::endl;
                ss << "float*  pblockA01 = (float*)(&blockA01);" << std::endl;
            }
        } else {
            ss << "float_t blockA00;" << std::endl;
            ss << "float*  pblockA00 = (float*)(&blockA00);" << std::endl;
            ss << "int pos = 0;" << std::endl;
            ss << "LOOP(KERNEL_WIDTH, pos," << std::endl;
            ss << "{" << std::endl;
            ss << "if (curr_y >= INPUT_PAD_H && "
               << "curr_y < input_height + INPUT_PAD_H && "
               << "curr_x + pos * DILATION_X >= INPUT_PAD_W && "
               << "curr_x + pos * DILATION_X < input_width + INPUT_PAD_W)"
               << std::endl;
            ss << "pblockA00[pos] = src0_read[pos * DILATION_X];" << std::endl;
            ss << "else" << std::endl;
            ss << "pblockA00[pos] = 0;" << std::endl;
            ss << "})" << std::endl;
            ss << "curr_y += DILATION_Y;" << std::endl;
            if (blockM == 2) {
                ss << "float_t blockA01;" << std::endl;
                ss << "float*  pblockA01 = (float*)(&blockA01);" << std::endl;
                ss << "pos = 0;" << std::endl;
                ss << "LOOP(KERNEL_WIDTH, pos," << std::endl;
                ss << "{" << std::endl;
                ss << "if (curr_y1 >= INPUT_PAD_H && "
                   << "curr_y1 < input_height + INPUT_PAD_H && "
                   << "curr_x1 + pos * DILATION_X >= INPUT_PAD_W && "
                   << "curr_x1 + pos * DILATION_X < input_width + INPUT_PAD_W)"
                   << std::endl;
                ss << "pblockA01[pos] = src0_read1[pos * DILATION_X];" << std::endl;
                ss << "else" << std::endl;
                ss << "pblockA01[pos] = 0;" << std::endl;
                ss << "})" << std::endl;
                ss << "curr_y1 += DILATION_Y;" << std::endl;
            }
        }
        ss << "src0_read += (ROW_PITCH * DILATION_Y);" << std::endl;
        if (blockM == 2) {
            ss << "src0_read1 += (ROW_PITCH * DILATION_Y);" << std::endl;
        }
        ss << "uint blockB00[KERNEL_WIDTH * (TILE_N_PER_LANE)];" << std::endl;
        ss << "float8* p8BlockB00 = (float8*)blockB00;" << std::endl;
        ss << "float4* p4BlockB00 = (float4*)blockB00;" << std::endl;
        ss << "float2* p2BlockB00 = (float2*)blockB00;" << std::endl;
        ss << "float*  pBlockB00 =  (float* )blockB00;" << std::endl;
        ss << "interleaved_y = 0;" << std::endl;
        ss << "LOOP(KERNEL_WIDTH_DIV2, interleaved_y, " << std::endl;
        ss << "{ " << std::endl;
        if (simd_size == 8) {
            ss << "p8BlockB00[interleaved_y] = as_float8("
               << "intel_sub_group_block_read8( (const __global uint*)src1_read ) ); "
               << std::endl;
        } else {
            ss << "p4BlockB00[interleaved_y] = as_float4("
               << "intel_sub_group_block_read4( (const __global uint*)src1_read ) ); "
               << std::endl;
        }
        ss << "src1_read += WIDTH1 * 2;" << std::endl;
        ss << "} )" << std::endl;
        ss << "if ( kernel_width_is_odd )" << std::endl;
        ss << "{" << std::endl;
        if (simd_size == 8) {
            ss << "p4BlockB00[KERNEL_WIDTH - 1] = as_float4("
               << "intel_sub_group_block_read4( (const __global uint*)src1_read ) ); "
               << std::endl;
        } else {
            ss << "p2BlockB00[KERNEL_WIDTH - 1] = as_float2("
               << "intel_sub_group_block_read2( (const __global uint*)src1_read ) ); "
               << std::endl;
        }
        ss << "src1_read += WIDTH1 * 2;" << std::endl;
        ss << "}" << std::endl;
        ss << "// Perform MADs" << std::endl;
        ss << "kernel_idx = 0;" << std::endl;
        ss << "interleaved_y = 0;" << std::endl;
        ss << "LOOP(KERNEL_WIDTH_DIV2, interleaved_y, " << std::endl;
        ss << "{" << std::endl;
        ss << "kernel_y = interleaved_y * 2;" << std::endl;
        if (simd_size == 16) {
            ss << "DOT_PRODUCT_16("
               << "blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); "
               << "kernel_idx++;"
               << std::endl;
            ss << "DOT_PRODUCT_16("
               << "blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); "
               << "kernel_idx++;"
               << std::endl;
            ss << "DOT_PRODUCT_16("
               << "blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); "
               << "kernel_idx++;"
               << std::endl;
            ss << "DOT_PRODUCT_16("
               << "blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); "
               << "kernel_idx++;"
               << std::endl;
        } else {
            ss << "DOT_PRODUCT_8( blockC00, pblockA00[kernel_y    ], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC00, pblockA00[kernel_y + 1], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC10, pblockA00[kernel_y    ], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC10, pblockA00[kernel_y + 1], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC20, pblockA00[kernel_y    ], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC20, pblockA00[kernel_y + 1], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC30, pblockA00[kernel_y    ], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC30, pblockA00[kernel_y + 1], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
        }
        if (blockM == 2) {
            ss << "kernel_idx -= 8;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC01, pblockA01[kernel_y    ], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC01, pblockA01[kernel_y + 1], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC11, pblockA01[kernel_y    ], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC11, pblockA01[kernel_y + 1], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC21, pblockA01[kernel_y    ], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC21, pblockA01[kernel_y + 1], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC31, pblockA01[kernel_y    ], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC31, pblockA01[kernel_y + 1], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
        }
        ss << "} )" << std::endl;
        ss << "kernel_y = interleaved_y * 2;" << std::endl;
        ss << "if ( kernel_width_is_odd )" << std::endl;
        ss << "{" << std::endl;
        if (simd_size == 16) {
            ss << "DOT_PRODUCT_16( blockC00, pblockA00[kernel_y], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_16( blockC10, pblockA00[kernel_y], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
        } else {
            ss << "DOT_PRODUCT_8( blockC00, pblockA00[kernel_y], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC10, pblockA00[kernel_y], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC20, pblockA00[kernel_y], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC30, pblockA00[kernel_y], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
        }
        if (blockM == 2) {
            ss << "kernel_idx -= 4;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC01, pblockA01[kernel_y], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC11, pblockA01[kernel_y], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC21, pblockA01[kernel_y], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC31, pblockA01[kernel_y], "
               << "pBlockB00[kernel_idx] ); kernel_idx++;" << std::endl;
        }
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "while( ++patch_row < KERNEL_HEIGHT );" << std::endl;
        // reset to start of next slice of patch
        ss << "src0_read += "
           << "SLICE_PITCH - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y);"
           << std::endl;
        if (blockM == 2) {
            // reset to start of next slice of patch
            ss << "src0_read1 += "
               << "SLICE_PITCH - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y);"
               << std::endl;
        }
        ss << "} " << std::endl;
        ss << "while ( ++patch_depth < INPUT_DEPTH );" << std::endl;
        // Dst resembles a cube of width x height x (output channel * batches).
        // Each tile writes: (SIMD * TILE_M) x 1 x TILE_N.
        // Partial writes most likely generated if padding used.
        ss << "__global float *out = dst " << std::endl;
        // batch offset
        ss << "+ global_z * OUT_PITCH_Z" << std::endl;
        // channel offset
        ss << "+ ( group_x * TILE_N ) * OUT_PITCH_Y" << std::endl;
        // y offset
        ss << "+ ( ( global_y * TILE_M ) / output_width + OUT_PADDING_HEIGHT) * "
            << "OUT_PITCH_X" << std::endl;
        // x offset
        ss << "+ ( ( global_y * TILE_M ) % output_width ) + OUT_PADDING_LEFT;"
            << std::endl;
        if (blockM == 2) {
            ss << "__global float *out1 = dst " << std::endl;
            ss << "+ global_z * OUT_PITCH_Z" << std::endl;
            ss << "+ ( group_x * TILE_N ) * OUT_PITCH_Y" << std::endl;
            ss << "+ ((global_y * TILE_M + 1) / output_width + OUT_PADDING_HEIGHT)*"
               << "OUT_PITCH_X" << std::endl;
            ss << "+ ( ( global_y * TILE_M + 1 ) % output_width ) + OUT_PADDING_LEFT;"
               << std::endl;
        }
        ss << "float bias[TILE_N_PER_LANE];" << std::endl;
        ss << "typedef CAT( float, TILE_N_PER_LANE) float_flex;" << std::endl;
        ss << "float_flex *bias_vec;" << std::endl;
        ss << "bias_vec = (float_flex*)bias;" << std::endl;
        if (simd_size == 16) {
            ss << "*bias_vec = "
               << "as_float2(intel_sub_group_block_read2("
               << "(__global uint *)biases + group_x * TILE_N));"
               << std::endl;
            // Work around a potential compiler bug
            ss << "if (group_x > 0xFFFFFFFEul)" << std::endl;
            ss << "out[0] = bias[0] + bias[1];" << std::endl;
        } else {
            ss << "*bias_vec = "
               << "as_float4(intel_sub_group_block_read4("
               << "(__global uint *)biases + group_x * TILE_N));"
               << std::endl;
        }
        ss << "if (global_y * TILE_M < output_width * output_height )" << std::endl;
        ss << "{" << std::endl;
        if (simd_size == 16) {
            ss << "for (int i = 0; i < 16; i++)" << std::endl;
            ss << "{" << std::endl;
            ss << "out[( 0+i) * OUT_PITCH_Y] = "
               << "blockC00[i] + intel_sub_group_shuffle(bias[0], i);" << std::endl;
            ss << "out[(16+i) * OUT_PITCH_Y] = "
               << "blockC10[i] + intel_sub_group_shuffle(bias[1], i);;" << std::endl;
            ss << "}" << std::endl;
        } else {
            ss << "for (int i = 0; i < 8; i++)" << std::endl;
            ss << "{" << std::endl;
            ss << "out[( 0+i) * OUT_PITCH_Y] = "
               << "blockC00[i] + intel_sub_group_shuffle(bias[0], i);" << std::endl;
            ss << "out[( 8+i) * OUT_PITCH_Y] = "
               << "blockC10[i] + intel_sub_group_shuffle(bias[1], i);" << std::endl;
            ss << "out[(16+i) * OUT_PITCH_Y] = "
               << "blockC20[i] + intel_sub_group_shuffle(bias[2], i);" << std::endl;
            ss << "out[(24+i) * OUT_PITCH_Y] = "
               << "blockC30[i] + intel_sub_group_shuffle(bias[3], i);" << std::endl;
            ss << "}" << std::endl;
        }
        if (blockM == 2) {
            ss << "if( global_y * TILE_M + 1 < output_width * output_height )"
               << std::endl;
            ss << "{" << std::endl;
            ss << "for( int i = 0; i < 8; i++ )" << std::endl;
            ss << "{" << std::endl;
            ss << "out1[( 0+i) * OUT_PITCH_Y] = "
               << "blockC01[i] + intel_sub_group_shuffle(bias[0], i);" << std::endl;
            ss << "out1[( 8+i) * OUT_PITCH_Y] = "
               << "blockC11[i] + intel_sub_group_shuffle(bias[1], i);" << std::endl;
            ss << "out1[(16+i) * OUT_PITCH_Y] = "
               << "blockC21[i] + intel_sub_group_shuffle(bias[2], i);" << std::endl;
            ss << "out1[(24+i) * OUT_PITCH_Y] = "
               << "blockC31[i] + intel_sub_group_shuffle(bias[3], i);" << std::endl;
            ss << "}" << std::endl;
            ss << "}" << std::endl;
        }
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "#if TILE_N_LAST > 0" << std::endl;
        ss << "else" << std::endl;
        ss << "{" << std::endl;
        // Result ctile (*dst) is M rows x N columns
        // LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.
        ss << "int i = 0;" << std::endl;
        ss << "float8  blockC[TILE_N_LAST_DIV8];" << std::endl;
        ss << "LOOP(TILE_N_LAST_DIV8, i," << std::endl;
        ss << "{" << std::endl;
        ss << "blockC[i] = 0.f;" << std::endl;
        ss << "} )" << std::endl;
        ss << "int curr_x = ( global_y % output_width ) * STRIDE_X;" << std::endl;
        ss << "int curr_y = ( global_y / output_width ) * STRIDE_Y;" << std::endl;
        if (pad_h_ != 0 || pad_w_ != 0 || dilation_w_ != 1 || dilation_h_ != 1) {
            ss << "int saved_y = curr_y;" << std::endl;
        }
        ss << "const __global float *src0_read = src0" << std::endl;
        ss << "+ ALIGNED_INPUT_SIZE * global_z" << std::endl;
        ss << "+ (curr_y - INPUT_PAD_H) * ROW_PITCH" << std::endl;
        ss << "+ (curr_x - INPUT_PAD_W);" << std::endl;
        if (blockM == 2) {
            ss << "i = 0;" << std::endl;
            ss << "float8  blockC1[TILE_N_LAST_DIV8];" << std::endl;
            ss << "LOOP(TILE_N_LAST_DIV8, i," << std::endl;
            ss << "{" << std::endl;
            ss << "blockC1[i] = 0.f;" << std::endl;
            ss << "} )" << std::endl;
            ss << "int curr_x1 = ((global_y * TILE_M + 1) % output_width) * STRIDE_X;"
               << std::endl;
            ss << "int curr_y1 = ((global_y * TILE_M + 1) / output_width) * STRIDE_Y;"
               << std::endl;
            if (pad_h_ != 0 || pad_w_ != 0 || dilation_w_ != 1 || dilation_h_ != 1) {
                ss << "int saved_y1 = curr_y1;" << std::endl;
            }
            ss << "const __global float *src0_read1 = src0" << std::endl;
            ss << "+ ALIGNED_INPUT_SIZE * global_z" << std::endl;
            ss << "+ (curr_y1 - INPUT_PAD_H) * ROW_PITCH" << std::endl;
            ss << "+ (curr_x1 - INPUT_PAD_W);" << std::endl;
        }
        ss << "const __global float *src1_read = src1 + ( global_x * TILE_N  * 2);"
           << std::endl;
        ss << "int patch_depth = 0;" << std::endl;
        ss << "do" << std::endl;
        ss << "{" << std::endl;
        ss << "int patch_row = 0;" << std::endl;
        if (pad_h_ != 0 || pad_w_ != 0 || dilation_w_ != 1 || dilation_h_ != 1) {
            ss << "curr_y = saved_y;" << std::endl;
            if (blockM == 2) {
                ss << "curr_y1 = saved_y1;" << std::endl;
            }
        }
        ss << "do" << std::endl;
        ss << "{" << std::endl;
        ss << "const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;"
           << std::endl;
        if (pad_h_ == 0 && pad_w_ == 0 && dilation_w_ == 1 && dilation_h_ == 1) {
            ss << "float_t blockA00 = ( (const __global float_t*)src0_read )[0];"
               << std::endl;
            ss << "float*  pblockA00 = (float*)(&blockA00);" << std::endl;
            if (blockM == 2) {
                ss << "float_t blockA01 = ( (const __global float_t*)src0_read1 )[0];"
                   << std::endl;
                ss << "float*  pblockA01 = (float*)(&blockA01);" << std::endl;
            }
        } else {
            ss << "float_t blockA00;" << std::endl;
            ss << "float*  pblockA00 = (float*)(&blockA00);" << std::endl;
            ss << "int pos = 0;" << std::endl;
            ss << "LOOP(KERNEL_WIDTH, pos," << std::endl;
            ss << "{" << std::endl;
            ss << "if (curr_y >= INPUT_PAD_H && "
               << "curr_y < input_height + INPUT_PAD_H && "
               << "curr_x + pos * DILATION_X >= INPUT_PAD_W && "
               << "curr_x + pos * DILATION_X < input_width + INPUT_PAD_W)"
               << std::endl;
            ss << "pblockA00[pos] = src0_read[pos * DILATION_X];" << std::endl;
            ss << "else" << std::endl;
            ss << "pblockA00[pos] = 0;" << std::endl;
            ss << "})" << std::endl;
            ss << "curr_y += DILATION_Y;" << std::endl;
            if (blockM == 2) {
                ss << "float_t blockA01;" << std::endl;
                ss << "float*  pblockA01 = (float*)(&blockA01);" << std::endl;
                ss << "pos = 0;" << std::endl;
                ss << "LOOP(KERNEL_WIDTH, pos," << std::endl;
                ss << "{" << std::endl;
                ss << "if (curr_y1 >= INPUT_PAD_H && "
                   << "curr_y1 < input_height + INPUT_PAD_H && "
                   << "curr_x1 + pos * DILATION_X >= INPUT_PAD_W && "
                   << "curr_x1 + pos * DILATION_X < input_width + INPUT_PAD_W)"
                   << std::endl;
                ss << "pblockA01[pos] = src0_read1[pos * DILATION_X];" << std::endl;
                ss << "else" << std::endl;
                ss << "pblockA01[pos] = 0;" << std::endl;
                ss << "})" << std::endl;
                ss << "curr_y1 += DILATION_Y;" << std::endl;
            }
        }
        ss << "src0_read += (ROW_PITCH * DILATION_Y);" << std::endl;
        if (blockM == 2) {
            ss << "src0_read1 += (ROW_PITCH * DILATION_Y);" << std::endl;
        }
        ss << "float blockB[KERNEL_WIDTH * TILE_N_LAST_DIV8];" << std::endl;
        ss << "interleaved_y = 0;" << std::endl;
        ss << "LOOP(KERNEL_WIDTH_DIV2, interleaved_y, " << std::endl;
        ss << "{ " << std::endl;
        ss << "#if TILE_N_LAST_DIV8 == 1" << std::endl;
        ss << "float2* p2BlockB = (float2* )blockB;" << std::endl;
        ss << "p2BlockB[interleaved_y] = as_float2("
           << "intel_sub_group_block_read2( (const __global uint*)src1_read ) );"
           << std::endl;
        ss << "#elif TILE_N_LAST_DIV8 == 2" << std::endl;
        ss << "float4* p4BlockB = (float4* )blockB;" << std::endl;
        ss << "p4BlockB[interleaved_y] = as_float4("
           << "intel_sub_group_block_read4( (const __global uint*)src1_read ) );"
           << std::endl;
        ss << "#elif TILE_N_LAST_DIV8 == 3" << std::endl;
        ss << "//TODO: broken.  No block_read6" << std::endl;
        ss << "float6* p6BlockB = (float6* )blockB;" << std::endl;
        ss << "(*((float8*)(&p6BlockB[interleaved_y]))).s0123 = as_float4("
           << "intel_sub_group_block_read4( (const __global uint*)src1_read ) );"
           << std::endl;
        ss << "(*((float8*)(&p6BlockB[interleaved_y]))).s45 = as_float2("
           << "intel_sub_group_block_read2("
           << "(const __global uint*)(src1_read + 4 * 8)));" << std::endl;
        ss << "#endif" << std::endl;
        ss << "src1_read += WIDTH1 * 2;" << std::endl;
        ss << "} )" << std::endl;
        ss << "if ( kernel_width_is_odd )" << std::endl;
        ss << "{" << std::endl;
        ss << "#if TILE_N_LAST_DIV8 == 1" << std::endl;
        ss << "float* pBlockB = (float* )blockB;" << std::endl;
        ss << "pBlockB[KERNEL_WIDTH - 1] = as_float("
           << "intel_sub_group_block_read( (const __global uint*)src1_read ) );"
           << std::endl;
        ss << "#elif TILE_N_LAST_DIV8 == 2" << std::endl;
        ss << "float2* p2BlockB = (float2* )blockB;" << std::endl;
        ss << "p2BlockB[KERNEL_WIDTH - 1] = as_float2("
           << "intel_sub_group_block_read2( (const __global uint*)src1_read ) );"
           << std::endl;
        ss << "#elif TILE_N_LAST_DIV8 == 3" << std::endl;
        ss << "float3* p3BlockB = (float3* )blockB;" << std::endl;
        ss << "p3BlockB[KERNEL_WIDTH - 1].s01 = as_float2("
           << "intel_sub_group_block_read2( (const __global uint*)src1_read ) );"
           << std::endl;
        ss << "p3BlockB[KERNEL_WIDTH - 1].s2 = as_float("
           << "intel_sub_group_block_read( (const __global uint*)"
           << "(src1_read + 2 * 8)));" << std::endl;
        ss << "#endif" << std::endl;
        ss << "src1_read += WIDTH1 * 2;" << std::endl;
        ss << "}" << std::endl;
        ss << "// Perform MADs" << std::endl;
        ss << "float* pBlockB = (float*)blockB;" << std::endl;
        ss << "kernel_idx = 0;" << std::endl;
        ss << "interleaved_y = 0;" << std::endl;
        ss << "LOOP(KERNEL_WIDTH_DIV2, interleaved_y, " << std::endl;
        ss << "{" << std::endl;
        ss << "kernel_y = interleaved_y * 2;" << std::endl;
        ss << "DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y    ],"
           << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
        ss << "DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y + 1],"
           << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
        ss << "#if TILE_N_LAST_DIV8 >= 2" << std::endl;
        ss << "DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y    ],"
           << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
        ss << "DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y + 1],"
           << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
        ss << "#if TILE_N_LAST_DIV8 >= 3" << std::endl;
        ss << "DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y    ],"
           << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
        ss << "DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y + 1],"
           << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
        ss << "#endif" << std::endl;
        ss << "#endif" << std::endl;
        if (blockM == 2) {
            ss << "kernel_idx -= TILE_N_LAST_DIV8 * 2;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y    ],"
               << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y + 1],"
               << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "#if TILE_N_LAST_DIV8 >= 2" << std::endl;
            ss << "DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y    ],"
               << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y + 1],"
               << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "#if TILE_N_LAST_DIV8 >= 3" << std::endl;
            ss << "DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y    ],"
               << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y + 1],"
               << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "#endif" << std::endl;
            ss << "#endif" << std::endl;
        }
        ss << "} )" << std::endl;
        ss << "kernel_y = interleaved_y * 2;" << std::endl;
        ss << "if ( kernel_width_is_odd )" << std::endl;
        ss << "{" << std::endl;
        ss << "DOT_PRODUCT_8( blockC[0], pblockA00[kernel_y],"
           << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
        ss << "#if TILE_N_LAST_DIV8 >= 2" << std::endl;
        ss << "DOT_PRODUCT_8( blockC[1], pblockA00[kernel_y],"
           << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
        ss << "#if TILE_N_LAST_DIV8 >= 3" << std::endl;
        ss << "DOT_PRODUCT_8( blockC[2], pblockA00[kernel_y],"
           << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
        ss << "#endif" << std::endl;
        ss << "#endif" << std::endl;
        if (blockM == 2) {
            ss << "kernel_idx -= TILE_N_LAST_DIV8;" << std::endl;
            ss << "DOT_PRODUCT_8( blockC1[0], pblockA01[kernel_y],"
               << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "#if TILE_N_LAST_DIV8 >= 2" << std::endl;
            ss << "DOT_PRODUCT_8( blockC1[1], pblockA01[kernel_y],"
               << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "#if TILE_N_LAST_DIV8 >= 3" << std::endl;
            ss << "DOT_PRODUCT_8( blockC1[2], pblockA01[kernel_y],"
               << "pBlockB[kernel_idx] ); kernel_idx++;" << std::endl;
            ss << "#endif" << std::endl;
            ss << "#endif" << std::endl;
        }
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "//while( ++patch_row < 1 ); //debug" << std::endl;
        ss << "while( ++patch_row < KERNEL_HEIGHT );" << std::endl;
        ss << "src0_read += "
           << "SLICE_PITCH - ( KERNEL_HEIGHT * ROW_PITCH * DILATION_Y );"
           << std::endl;
        ss << "} " << std::endl;
        ss << "while ( ++patch_depth < INPUT_DEPTH );" << std::endl;
        ss << "__global float *out = dst " << std::endl;
        ss << "+ global_z * OUT_PITCH_Z" << std::endl;
        ss << "+ (group_x * TILE_N) * OUT_PITCH_Y" << std::endl;
        ss << "+ ((global_y * TILE_M) / output_width + "
           << "OUT_PADDING_HEIGHT) * OUT_PITCH_X" << std::endl;
        ss << "+ ((global_y * TILE_M) % output_width ) + OUT_PADDING_LEFT;"
           << std::endl;
        if (blockM == 2) {
            ss << "__global float *out1 = dst " << std::endl;
            ss << "+ global_z * OUT_PITCH_Z" << std::endl;
            ss << "+ ( group_x * TILE_N ) * OUT_PITCH_Y" << std::endl;
            ss << "+ ((global_y * TILE_M + 1) / output_width + OUT_PADDING_HEIGHT ) *"
                << "OUT_PITCH_X" << std::endl;
            ss << "+ ((global_y * TILE_M + 1) % output_width ) + OUT_PADDING_LEFT;"
                << std::endl;
        }
        ss << "float bias[4];" << std::endl;
        ss << "float4 *bias_vec;" << std::endl;
        ss << "bias_vec = (float4*)bias;" << std::endl;
        ss << "*bias_vec = as_float4(intel_sub_group_block_read4("
           << "(__global uint *)biases + group_x * TILE_N));" << std::endl;
        ss << "if (global_y * TILE_M < output_width * output_height )" << std::endl;
        ss << "{" << std::endl;
        ss << "for (int i = 0; i < 8; i++)" << std::endl;
        ss << "{" << std::endl;
        ss << "if ( TILE_N_LAST_DIV8 > 0 ) out[( 0+i) * OUT_PITCH_Y] = "
           << "blockC[0][i] + intel_sub_group_shuffle(bias[0], i);" << std::endl;
        ss << "if ( TILE_N_LAST_DIV8 > 1 ) out[( 8+i) * OUT_PITCH_Y] = "
           << "blockC[1][i] + intel_sub_group_shuffle(bias[1], i);" << std::endl;
        ss << "if ( TILE_N_LAST_DIV8 > 2 ) out[(16+i) * OUT_PITCH_Y] = "
           << "blockC[2][i] + intel_sub_group_shuffle(bias[2], i);" << std::endl;
        ss << "if ( TILE_N_LAST_DIV8 > 3 ) out[(24+i) * OUT_PITCH_Y] = "
           << "blockC[3][i] + intel_sub_group_shuffle(bias[3], i);" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        if (blockM == 2) {
            ss << "if( global_y * TILE_M + 1 < output_width * output_height )"
               << std::endl;
            ss << "{" << std::endl;
            ss << "for( int i = 0; i < 8; i++ )" << std::endl;
            ss << "{" << std::endl;
            ss << "if ( TILE_N_LAST_DIV8 > 0 ) out1[( 0+i) * OUT_PITCH_Y] = "
               << "blockC1[0][i] + intel_sub_group_shuffle(bias[0], i);" << std::endl;
            ss << "if ( TILE_N_LAST_DIV8 > 1 ) out1[( 8+i) * OUT_PITCH_Y] = "
               << "blockC1[1][i] + intel_sub_group_shuffle(bias[1], i);" << std::endl;
            ss << "if ( TILE_N_LAST_DIV8 > 2 ) out1[(16+i) * OUT_PITCH_Y] = "
               << "blockC1[2][i] + intel_sub_group_shuffle(bias[2], i);" << std::endl;
            ss << "if ( TILE_N_LAST_DIV8 > 3 ) out1[(24+i) * OUT_PITCH_Y] = "
               << "blockC1[3][i] + intel_sub_group_shuffle(bias[3], i);" << std::endl;
            ss << "}" << std::endl;
            ss << "}" << std::endl;
        }
        ss << "}" << std::endl;
        ss << "#endif" << std::endl;
        ss << "}" << std::endl;
    } else if (kernelType == KERNEL_TYPE_BASIC) {
        kernelUKey = generate_specific_key(4, blockM, blockK, blockN);
        kernel_name_ = "BASIC_";
        kernel_name_ += kernelUKey.c_str();

        // opts
        opts << " -cl-fast-relaxed-math -D CFMultiNoPadding=" << kernel_name_;
        if (IsBeignet())
            opts << " -D__BEIGNET__ ";
        else
            opts << " -cl-no-subgroup-ifp ";
        options_ = opts.str();

        // defs
        add_def(ss, "CHANNELS", channels_ / group_);
        add_def(ss, "APPLY_BIAS", bias_term_);
        add_def(ss, "OUTPUT_Z", M_);
        add_def(ss, "ZPAR", 1);

        // kernel
        ss << "#define ACTIVATION_FUNCTION(_dst_, _offset_, _data_) "
           << "do { (_dst_)[(_offset_)] = (_data_);} while(0)" << std::endl;
        ss << "__kernel void CFMultiNoPadding(" << std::endl;
        ss << "__global Dtype* image_data," << std::endl;
        ss << "int_tp image_offset," << std::endl;
        ss << "__global Dtype* kernel_data, " << std::endl;
        ss << "int_tp kernel_offset," << std::endl;
        ss << "__global Dtype* bias," << std::endl;
        ss << "const int_tp bias_offset," << std::endl;
        ss << "__global Dtype* convolved_image, " << std::endl;
        ss << "const int_tp convolved_image_offset," << std::endl;
        ss << "const ushort input_width," << std::endl;
        ss << "const ushort input_height," << std::endl;
        ss << "const ushort output_width," << std::endl;
        ss << "const ushort output_height," << std::endl;
        ss << "const ushort pad_w," << std::endl;
        ss << "const ushort pad_h) {" << std::endl;
        ss << "const int_tp outputX = get_global_id(0);" << std::endl;
        ss << "const int_tp outputY = get_global_id(1);" << std::endl;
        ss << "const int_tp kernelNum = get_global_id(2)*ZPAR;" << std::endl;
        ss << "if(outputX < output_width && outputY < output_height)" << std::endl;
        ss << "{" << std::endl;
        ss << "Dtype sum[ZPAR];" << std::endl;
        ss << "for(int_tp kern =0; kern < ZPAR; kern++)" << std::endl;
        ss << "{" << std::endl;
        ss << "sum[kern] = 0.0f;" << std::endl;
        ss << "}" << std::endl;
        ss << "const int_tp org_y = outputY * STRIDE_Y - pad_h;" << std::endl;
        ss << "const int_tp org_x = outputX * STRIDE_X - pad_w;" << std::endl;
        ss << "const int_tp currentKernelOffset = "
           << "kernel_offset + kernelNum*KERNEL_HEIGHT*KERNEL_WIDTH*CHANNELS;"
           << std::endl;
        ss << "const int_tp biasIndex=bias_offset + kernelNum;" << std::endl;
        ss << "const int_tp local_image_offset = org_y*input_width + org_x;"
           << std::endl;
        ss << "const int_tp imageSize = input_width*input_height;" << std::endl;
        ss << "__global Dtype* image_dataPtrFloat = "
           << "(image_data + (image_offset + local_image_offset));" << std::endl;
        ss << "__global Dtype* kernel_dataPtrFloat = "
           << "(kernel_data + (currentKernelOffset));" << std::endl;
        ss << "for(int_tp c = 0; c < CHANNELS; c++)" << std::endl;
        ss << "{" << std::endl;
        ss << "for(int_tp y = 0; y < KERNEL_HEIGHT; y++)" << std::endl;
        ss << "{" << std::endl;
        ss << "for(int_tp x = 0; x < KERNEL_WIDTH; x++)" << std::endl;
        ss << "{" << std::endl;
        ss << "if(!(org_y + y * DILATION_Y >= 0 && "
           << "org_y + y * DILATION_Y < input_height && "
           << "org_x + x * DILATION_X >= 0 && "
           << "org_x + x * DILATION_X < input_width))" << std::endl;
        ss << "{" << std::endl;
        ss << "continue;" << std::endl;
        ss << "}" << std::endl;
        ss << "for(int_tp kern =0; kern < ZPAR; kern++)" << std::endl;
        ss << "{" << std::endl;
        ss << "sum[kern] += image_dataPtrFloat[x * DILATION_X] * "
           << "kernel_dataPtrFloat[kern*KERNEL_HEIGHT*KERNEL_WIDTH*CHANNELS + x];"
           << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "image_dataPtrFloat += input_width * DILATION_Y;" << std::endl;
        ss << "kernel_dataPtrFloat += KERNEL_WIDTH;" << std::endl;
        ss << "}" << std::endl;
        ss << "image_dataPtrFloat += "
           << "imageSize - input_width*KERNEL_HEIGHT*DILATION_Y;" << std::endl;
        ss << "}" << std::endl;
        ss << "if(APPLY_BIAS == 1)" << std::endl;
        ss << "{" << std::endl;
        ss << "for(int_tp kern = 0; kern < ZPAR; kern++)" << std::endl;
        ss << "{" << std::endl;
        ss << "if(kernelNum+kern < OUTPUT_Z)" << std::endl;
        ss << "{" << std::endl;
        ss << "int_tp offset = convolved_image_offset + "
           << "(kernelNum+kern)*output_height*output_width + "
           << "outputY*output_width + outputX;" << std::endl;
        ss << "ACTIVATION_FUNCTION(convolved_image, offset, sum[kern] + "
           << "bias[biasIndex +kern]);" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "else" << std::endl;
        ss << "{" << std::endl;
        ss << "for(int_tp kern = 0; kern < ZPAR; kern++)" << std::endl;
        ss << "{" << std::endl;
        ss << "if(kernelNum+kern < OUTPUT_Z)" << std::endl;
        ss << "{" << std::endl;
        ss << "int_tp offset = convolved_image_offset + "
           << "(kernelNum+kern)*output_height*output_width + "
           << "outputY*output_width + outputX;" << std::endl;
        ss << "ACTIVATION_FUNCTION(convolved_image, offset, sum[kern]);"
           << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
        ss << "}" << std::endl;
    }
    return ss.str();
}

template<typename Dtype>
void LibDNNConvSpatial<Dtype>::GenerateKernels()
{
    std::stringstream ss;

    ss << generate_header();
    ss << generate_fw_defs();
    ss << generate_fw_kernels(kernelType_, blockM_, blockK_, blockN_);

    kernel_ = ss.str();
}

template<typename Dtype>
bool LibDNNConvSpatial<Dtype>::Forward(const Dtype* bottom_data,
                                       const Dtype* weight,
                                       const Dtype* bias, Dtype* top_data,
                                       int_tp batch_size)
{
    weight_ = weight;
    if (bias_term_)
        bias_ = bias;
    bottom_data_ = bottom_data;
    top_data_ = top_data;
    bias_offset_ = 0;
    num_ = batch_size;

    if (!try_cache_) {
        load_cached_kernels(bottom_data, top_data);
        try_cache_ = true;
    }

    if (!tuned_)
        Tune(top_data, NULL, weight, NULL, bias, NULL,
             bottom_data, NULL, batch_size);

    cl_int ret = convolve(bottom_data, top_data, 0, num_, bestKernelConfig);
    return ret == CL_SUCCESS ? true : false;
}

template<typename Dtype>
void LibDNNConvSpatial<Dtype>::Tune(Dtype* top_data, Dtype* top_diff,
                                    const Dtype* weight,
                                    Dtype* weight_diff,
                                    const Dtype* bias,
                                    Dtype* bias_diff,
                                    const Dtype* bottom_data,
                                    Dtype* bottom_diff,
                                    int_tp batch_size)
{
    cl_int err;
    Dtype *verify_data;
    ocl::Context &ctx = ocl::Context::getDefault();

    verify_data =
        reinterpret_cast<Dtype*>(clCreateBuffer((cl_context)ctx.ptr(),
                                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                                batch_size * fmaps_out_ * out_spatial_dim_ * sizeof(Dtype),
                                                NULL, &err));
    CHECK_EQ(err, CL_SUCCESS) << "Failed to create verify buffer." << std::endl;

    calculate_verify_data(bottom_data, weight, bias, verify_data);
    setup_convolution(bottom_data, top_data, verify_data);
    clReleaseMemObject((cl_mem)verify_data);
    CHECK_EQ(tuned_, true) << "Spatial convolution auto-tuning failed.";
}

template<typename Dtype>
void LibDNNConvSpatial<Dtype>::calculate_verify_data(const Dtype* bottom,
                                                     const Dtype* w,
                                                     const Dtype* bias,
                                                     Dtype* verify_data)
{
    create_basic_kernel(bottom, verify_data, 1, 1, 1);
    kernel_index_ = kernelQueue.size() - 1;
    convolve(bottom, verify_data, 0, num_, kernelQueue[kernel_index_]);
    ocl::Queue &queue = ocl::Queue::getDefault();
    clEnqueueCopyBuffer((cl_command_queue)queue.ptr(),
                        (cl_mem)top_data_,
                        (cl_mem)verify_data, 0, 0,
                        sizeof(float) * num_ * this->top_dim_, 0, NULL, NULL);
    phash.erase(kernelQueue[kernel_index_]->kernelName);
    kernelQueue.pop_back();
    return;
}

template<typename Dtype>
void LibDNNConvSpatial<Dtype>::ForwardBenchmark(const Dtype* bottom,
                                                const Dtype* w,
                                                const Dtype* bias,
                                                Dtype* top,
                                                int_tp batch_size)
{
    weight_ = w;
    if (bias_term_)
        bias_ = bias;
    bottom_data_ = bottom;
    top_data_ = top;
    bias_offset_ = 0;
    num_ = batch_size;
    calculate_verify_data(bottom, w, bias, top);
}

#define dbg
#ifdef dbg
#define dbgPrint(x) (x)
#else
#define dbgPrint(x)
#endif

// For large enough input size, we do not need to tune kernels for different
// size. The reason is with large input size, there will be enough work items
// to feed al the EUs.
// FIXME for the gemm like convolution, switch back to eaxct image size.

#define TUNING_SIZE(x) ((x) > 256 ? 256 : (ALIGN(x, 16)))

// Computes 64-bit "cyclic redundancy check" sum, as specified in ECMA-182
template<typename Dtype>
uint64 LibDNNConvSpatial<Dtype>::crc64(const uchar* data, size_t size, uint64 crc0)
{
    static uint64 table[256];
    static bool initialized = false;

    if( !initialized )
    {
        for( int i = 0; i < 256; i++ )
        {
            uint64 c = i;
            for( int j = 0; j < 8; j++ )
                c = ((c & 1) ? CV_BIG_UINT(0xc96c5795d7870f42) : 0) ^ (c >> 1);
            table[i] = c;
        }
        initialized = true;
    }

    uint64 crc = ~crc0;
    for( size_t idx = 0; idx < size; idx++ )
        crc = table[(uchar)crc ^ data[idx]] ^ (crc >> 8);

    return ~crc;
}

template<typename Dtype>
void LibDNNConvSpatial<Dtype>::generate_key()
{
    std::stringstream keyBuilder;
    // FIXME: to support fuse?
    keyBuilder << kernel_w_ << "_"
               << kernel_h_ << "_"
               << channels_ << "_"
               << group_ << "_"
               << stride_h_ << "_"
               << stride_w_ << "_"
               << dilation_h_ << "_"
               << dilation_w_ << "_"
               << bias_term_ << "_"
               << TUNING_SIZE(width_) << "_"
               << TUNING_SIZE(height_) << "_"
               << pad_w_ << "_"
               << pad_h_ << "_"
               << num_ << "_"
               << M_;

    std::string prefix = ocl::Device::getDefault().name() +
                         ocl::Device::getDefault().vendorName() +
                         ocl::Device::getDefault().driverVersion() +
                         std::to_string(ocl::Device::getDefault().maxComputeUnits());
    prefix = prefix + keyBuilder.str();
    key_ = std::to_string(crc64((uchar*)prefix.c_str(), prefix.size()));
    short_key_ = keyBuilder.str();
}

template<typename Dtype>
std::string LibDNNConvSpatial<Dtype>::generate_specific_key(int_tp type, int_tp blockWidth,
	                                                    int_tp blockHeight, int_tp blockDepth)
{
    std::stringstream keyBuilder;
    keyBuilder << short_key_
               << "_" << type
               << "_" << blockWidth
               << "_" << blockHeight
               << "_" << blockDepth;
    return keyBuilder.str();
}

template<typename Dtype>
void interleaveMatrix(Dtype* mem_dst, const Dtype *mem,
                      int r, int c, int interleavedRows, int nonInterleavedRows,
                      int blockWidth, int rowAlignment )
{
    CHECK_EQ(interleavedRows % 2, 0) <<
             "interleaveMatrix only supports even values for interleavedRows.";

    size_t memSize = r * c * sizeof(float);
    size_t dstSize = memSize *
                     (interleavedRows + nonInterleavedRows * 2) /
                     (interleavedRows + nonInterleavedRows);
    memset(mem_dst, 0, dstSize);    // NOLINT

    const int xStride = blockWidth;
    const int yStride = c * 2;
    const Dtype *pSrc = mem;
    Dtype* pDst = mem_dst;
    for (int y = 0; y < r;) {
        for (int rows = 0; rows < interleavedRows; rows += 2) {
            if ( y >= r ) break;
            if ((c % xStride) == 0) {
                for (int x = 0; x < c / xStride; x++) {
                    memcpy(pDst + x * xStride * 2,                         // NOLINT
                           pSrc + x * xStride,     xStride * sizeof(Dtype));
                    memcpy(pDst + x * xStride * 2 + xStride,               // NOLINT
                           pSrc + x * xStride + c, xStride * sizeof(Dtype));
                }
            } else {
                const int count = c / xStride;
                int x = 0;
                for (; x < count - 1; x++) {
                    memcpy(pDst + x * xStride * 2,                          // NOLINT
                           pSrc + x * xStride, xStride * sizeof(Dtype));
                    memcpy(pDst + x * xStride * 2 + xStride,                // NOLINT
                           pSrc + x * xStride + c, xStride * sizeof(Dtype));
                }
                memcpy(pDst + x * xStride * 2,                            // NOLINT
                       pSrc + x * xStride, xStride * sizeof(Dtype));
            }
            pSrc += yStride;
            pDst += yStride;
            y += 2;
        }

        for (int rows = 0; rows < nonInterleavedRows; rows++) {
            if (y >= r) break;
            const int stride = rowAlignment;
            int remaining = c;
            for (int x = 0; x < c; x += stride) {
                if (remaining >= stride) {
                    memcpy(pDst + x * 2, pSrc + x, stride * sizeof(Dtype));    // NOLINT
                    remaining -=stride;
                } else {
                    memcpy(pDst + x * 2, pSrc + x, remaining * sizeof(Dtype));  // NOLINT
                }
            }
            pSrc += yStride / 2;
            pDst += yStride;
            y++;
        }
    }
}

template<typename Dtype>
void LibDNNConvSpatial<Dtype>::swizzleWeights(const Dtype *bottom,
                                              const Dtype *top,
                                              int_tp swizzled_factor,
                                              bool interleave)
{
    // Simply skip the weight swizzle if we already got a swizzled_weights_
    // in test phase and not in auto tuning
    // This requires we always call convolve again with the winner configuration
    // during the auto tuning stage.
    if (tuned_ &&
        swizzled_weights_ != NULL &&
        phase_test_ == true)
        return;

    cl_int err;
    ocl::Context ocl_ctx = ocl::Context::getDefault();
    if (swizzled_weights_ == NULL) {
        swizzled_weights_ = reinterpret_cast<Dtype*>(clCreateBuffer((cl_context)ocl_ctx.ptr(),
                                                                    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                                                    sizeof(Dtype) *
                                                                    ((num_output_ + 15) & ~15) *
                                                                    channels_ * kernel_h_ * ((kernel_w_ + 1) & ~1),
                                                                    NULL, &err));
        CHECK_EQ(err, CL_SUCCESS) << "Failed to create swizzled_weights buffer.";
    }

    ocl::Queue queue = ocl::Queue::getDefault();

    if (!interleave) {
        ocl::Kernel oclk_copy_weight(CL_KERNEL_SELECT("copyWeightsSwizzled"), cv::ocl::dnn::conv_spatial_helper_oclsrc);
        cl_uint argIdx = 0;

        int_tp channels = this->channels_ / this->group_;
        oclk_copy_weight.set(argIdx++, (cl_mem) weight_);
        oclk_copy_weight.set(argIdx++, (cl_mem) swizzled_weights_);
        oclk_copy_weight.set(argIdx++, kernel_w_);
        oclk_copy_weight.set(argIdx++, kernel_h_);
        oclk_copy_weight.set(argIdx++, channels);
        oclk_copy_weight.set(argIdx++, this->num_output_);
        oclk_copy_weight.set(argIdx++, swizzled_factor);
        const size_t global_work_size_Copy[3] = {
            (size_t) (ALIGN(this->num_output_, swizzled_factor) * channels * kernel_w_ * kernel_h_), 1, 1 };

        OCL_CHECK(clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                         (cl_kernel)oclk_copy_weight.ptr(), 3, NULL,
                                         global_work_size_Copy, NULL, 0, NULL,
                                         NULL));
    } else {
        Dtype* cpu_weight =
            reinterpret_cast<Dtype*>(clEnqueueMapBuffer((cl_command_queue)queue.ptr(),
                                                        (cl_mem)weight_, true, CL_MAP_READ, 0,
                                                        sizeof(Dtype) * num_output_ * kernel_dim_ * group_,
                                                        0, NULL, NULL, NULL));

        // assumption: kernel dimesion is 2
        Dtype* cpu_swizzled_weight =
            reinterpret_cast<Dtype*>(clEnqueueMapBuffer((cl_command_queue)queue.ptr(),
                                                        (cl_mem)swizzled_weights_,
                                                        true, CL_MAP_WRITE, 0,
                                                        sizeof(Dtype) *
                                                        ((num_output_ + 15) & ~15) *
                                                        channels_ * kernel_h_ * ((kernel_w_ + 1) & ~1),
                                                        0, NULL, NULL, NULL));

        int interleavedRows = (kernel_w_ / 2) * 2;
        int nonInterleavedRows = kernel_w_ % 2;
        int blockWidth = swizzled_factor;  // should equal to simd size.
        int rowAlignment = 32;
        size_t interleaved_filter_size = M_ * kernel_w_ * kernel_h_ * channels_ * sizeof(Dtype);
        Dtype * tmpSwizzledWeight = reinterpret_cast<Dtype*>(malloc(interleaved_filter_size));
        CHECK_EQ(tmpSwizzledWeight != NULL, true) << "Failed to allocate temporary swizzled weight";
        for (int od = 0; od < M_; od++)
            for (int id = 0; id < channels_; id++)
                for (int r = 0; r < kernel_h_; r++)
                    for (int c = 0; c < kernel_w_; c++)
                        tmpSwizzledWeight[((id * kernel_h_ + r)* kernel_w_ + c) * M_ + od] =
                            cpu_weight[((od * channels_ + id) * kernel_h_ + r)*kernel_w_+c];
        interleaveMatrix(cpu_swizzled_weight,
                         tmpSwizzledWeight,
                         kernel_w_ * kernel_h_ * channels_, M_,
                         interleavedRows,
                         nonInterleavedRows,
                         blockWidth,
                         rowAlignment);

        clEnqueueUnmapMemObject((cl_command_queue)queue.ptr(),
                                (cl_mem)weight_,
                                cpu_weight, 0, NULL,
                                NULL);
        clEnqueueUnmapMemObject((cl_command_queue)queue.ptr(),
                                (cl_mem)swizzled_weights_,
                                cpu_swizzled_weight, 0, NULL,
                                NULL);
        free(tmpSwizzledWeight);
    }
}

template<>
void LibDNNConvSpatial<float>::calculate_global_size(int_tp batch,
                                                     int_tp* wio,    // work item output size
                                                     size_t* lSize,  // local size
                                                     size_t* gSize)  // global size
{  
    gSize[0] = ceil((fmax(static_cast<float>(output_w_) / wio[0], 1.0)) / lSize[0]) * lSize[0];
    gSize[1] = ceil((fmax(static_cast<float>(output_h_) / wio[1], 1.0)) / lSize[1]) * lSize[1];
    gSize[2] = ceil(static_cast<float>((ceil(static_cast<float>(M_) * batch / wio[2]))) / lSize[2]) * lSize[2];
}

template<>
bool LibDNNConvSpatial<float>::create_basic_kernel(const float *bottom, const float *top,
                                                   int_tp blockWidth,
                                                   int_tp blockHeight, int_tp blockDepth)
{
    int_tp workItemOutput[3];
    workItemOutput[0] = 1;
    workItemOutput[1] = 1;
    workItemOutput[2] = 1;

    kernelType_ = 4;
    blockM_ = blockWidth;
    blockK_ = blockHeight;
    blockN_ = blockDepth;
    GenerateKernels();
    compile_fw_kernel();

    size_t localSize[3] = { 1, 1, 1 };
    size_t globalSize[3];

    calculate_global_size(1, workItemOutput, localSize, globalSize);
    kernelQueue.push_back(new kernelConfig(kernel_name_, globalSize, localSize, workItemOutput,
                                           false, false, true, 4));

    return true;
}

template<typename Dtype>
void LibDNNConvSpatial<Dtype>::setBufferKernelArg(const Dtype *bottom, const Dtype *top,
                                                  ocl::Kernel *kernel,
                                                  const cl_uint &argIdx,
                                                  ocl::Context *ctx,
                                                  cl_mem buffer, size_t offset,
                                                  size_t size, bool readOnly,
                                                  bool preserved)
{

    if (offset == 0) {
        kernel->set(argIdx, (cl_mem)buffer);
        return;
    }

    if (preserved &&
        subBufferMap.find(std::make_tuple(buffer, offset, size)) != subBufferMap.end()) {
        kernel->set(argIdx, (cl_mem)(subBufferMap.find(std::make_tuple(buffer, offset, size))->second));
        return;
    }
    cl_buffer_region region;
    region.origin = offset * sizeof(Dtype);
    region.size = size * sizeof(Dtype);
    cl_mem_flags memFlags = readOnly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
    cl_int error;
    cl_mem sub_buffer = clCreateSubBuffer(buffer, memFlags,
                                          CL_BUFFER_CREATE_TYPE_REGION,
                                          &region, &error);
    CHECK_EQ(error, CL_SUCCESS) << "Failed to create sub buffer." << std::endl;
    if (error != CL_SUCCESS) {
        dbgPrint(std::cout << "Failed to create sub buffer (" << error << ")." << std::endl);
        throw(error);
    }
    kernel->set(argIdx, (cl_mem)sub_buffer);
    if (preserved)
        subBufferMap.insert(std::make_pair(std::make_tuple(buffer, offset, size),
                            sub_buffer));
    else
        tmpSubBuffers.push_back(sub_buffer);
}

template<typename Dtype>
void LibDNNConvSpatial<Dtype>::cleanTmpSubBuffers(const Dtype *bottom, const Dtype *top)
{
    for (auto &buffer : tmpSubBuffers)
        clReleaseMemObject(buffer);
    tmpSubBuffers.clear();
}

template<>
cl_int LibDNNConvSpatial<float>::convolve(const float *bottom, const float *top,
                                          int_tp index,
                                          int_tp numImages, kernelConfig* config)
{
    ocl::Context ctx = ocl::Context::getDefault();
    ocl::Program program;
    phash_t::iterator it = phash.find(config->kernelName);
    if (it != phash.end())
        program = it->second;
    else
        return CL_INVALID_PROGRAM;
    ocl::Kernel kernel(config->kernelName.c_str(), program);
    cl_int err = CL_SUCCESS;

    if (config->kernelType == 2) {
        swizzleWeights(bottom, top, config->workItem_output[2], false);
        size_t total_bottom_size = bottom_dim_ * numImages;
        size_t total_kernel_size = kernel_h_ * kernel_w_ * channels_ * M_;
        size_t total_bias_size = M_ * group_;
        size_t total_top_size = top_dim_ * numImages;
        for (int_tp g = 0; g < group_; ++g) {
            bias_offset_ = M_ * g;
            int_tp image_offset = width_ * height_ * (channels_ / group_) * g;
            int_tp output_image_offset = output_w_ * output_h_ * M_ * g;

            int_tp kernel_offset = kernel_h_ * kernel_w_ * (channels_ / group_) * M_ * g;
            cl_uint argIdx = 0;

            try {
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) bottom_data_,
                                   image_offset,
                                   total_bottom_size - image_offset,
                                   true, false);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) swizzled_weights_,
                                   kernel_offset,
                                   total_kernel_size - kernel_offset,
                                   true, true);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) bias_,
                                   bias_offset_,
                                   total_bias_size - bias_offset_,
                                   true, true);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) top_data_,
                                   output_image_offset,
                                   total_top_size - output_image_offset,
                                   false, false);
            } catch (int e) {
                err = e;
            }

            if (err == CL_SUCCESS) {
                kernel.set(argIdx++, (uint16_t)width_);
                kernel.set(argIdx++, (uint16_t)height_);
                kernel.set(argIdx++, (uint16_t)output_w_);
                kernel.set(argIdx++, (uint16_t)output_h_);
                ocl::Queue queue = ocl::Queue::getDefault();
                err = clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                             (cl_kernel)kernel.ptr(), 3,
                                             NULL,
                                             config->global_work_size,
                                             config->local_work_size, 0, NULL,
                                             NULL);
            }
            if (err != CL_SUCCESS)
                break;
        }

        if (group_ > 1) {
            cleanTmpSubBuffers(bottom, top);
        }
        if (err != CL_SUCCESS)
            return err;
    } else if (config->kernelType == 5) {
        swizzleWeights(bottom, top, config->workItem_output[1], true);
        size_t total_bottom_size = bottom_dim_ * numImages;
        size_t total_kernel_size = kernel_h_ * kernel_w_ * channels_ * M_;
        size_t total_bias_size = M_ * group_;
        size_t total_top_size = top_dim_ * numImages;
        for (int_tp g = 0; g < group_; ++g) {
            bias_offset_ = M_ * g;
            int_tp image_offset = width_ * height_ * (channels_ / group_) * g;
            int_tp output_image_offset = output_w_ * output_h_ * M_ * g;

            cl_uint argIdx = 0;
            int_tp kernel_offset = kernel_h_ * kernel_w_ * (channels_ / group_) * M_ * g;
            try {
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) bottom_data_,
                                   image_offset,
                                   total_bottom_size - image_offset,
                                   true, false);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) swizzled_weights_,
                                   kernel_offset,
                                   total_kernel_size - kernel_offset,
                                   true, true);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) bias_,
                                   bias_offset_,
                                   total_bias_size - bias_offset_,
                                   true, true);
                setBufferKernelArg(bottom, top, &kernel, argIdx++, &ctx,
                                   (cl_mem) top_data_,
                                   output_image_offset,
                                   total_top_size - output_image_offset,
                                   false, false);
            } catch (int e) {
                err = e;
            }

            if (err == CL_SUCCESS) {
                kernel.set(argIdx++, (uint16_t)width_);
                kernel.set(argIdx++, (uint16_t)height_);
                kernel.set(argIdx++, (uint16_t)output_w_);
                kernel.set(argIdx++, (uint16_t)output_h_);
                ocl::Queue queue = ocl::Queue::getDefault();
                err = clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                             (cl_kernel)kernel.ptr(), 3,
                                             NULL,
                                             config->global_work_size,
                                             config->local_work_size, 0, NULL,
                                             NULL);
                OCL_CHECK(err);
            }
            if (err != CL_SUCCESS)
                break;
        }

        if (group_ > 1) {
            cleanTmpSubBuffers(bottom, top);
        }
        if (err != CL_SUCCESS)
            return err;
    } else {
        for (int_tp n = 0; n < numImages; ++n) {
            for (int_tp g = 0; g < group_; ++g) {
                bias_offset_ = M_ * g;
                int_tp image_offset = n * this->bottom_dim_
                    + width_ * height_ * (channels_ / group_) * g;
                int_tp output_image_offset = n * this->top_dim_
                    + output_w_ * output_h_ * M_ * g;

                cl_uint argIdx = 0;
                int_tp kernel_offset = kernel_h_ * kernel_w_ * (channels_ / group_) * M_ * g;

                kernel.set(argIdx++, (cl_mem) bottom_data_);
                kernel.set(argIdx++, image_offset);
                kernel.set(argIdx++, (cl_mem) weight_);
                kernel.set(argIdx++, kernel_offset);
                kernel.set(argIdx++, (cl_mem) bias_);
                kernel.set(argIdx++, bias_offset_);
                kernel.set(argIdx++, (cl_mem) top_data_);
                kernel.set(argIdx++, output_image_offset);
                kernel.set(argIdx++, (uint16_t)width_);
                kernel.set(argIdx++, (uint16_t)height_);
                kernel.set(argIdx++, (uint16_t)output_w_);
                kernel.set(argIdx++, (uint16_t)output_h_);
                kernel.set(argIdx++, (uint16_t)pad_w_);
                kernel.set(argIdx++, (uint16_t)pad_h_);
                ocl::Queue queue = ocl::Queue::getDefault();
                if (config->use_null_local) {
                    err = clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                                 (cl_kernel)kernel.ptr(), 3,
                                                 NULL,
                                                 config->global_work_size, NULL, 0, NULL,
                                                 NULL);
                } else {
                    err = clEnqueueNDRangeKernel((cl_command_queue)queue.ptr(),
                                                 (cl_kernel)kernel.ptr(), 3,
                                                 NULL,
                                                 config->global_work_size,
                                                 config->local_work_size, 0, NULL,
                                                 NULL);
                }

                if (err != CL_SUCCESS)
                    return err;
            }
        }
    }

    return err;
}

template<>
float LibDNNConvSpatial<float>::timed_convolve(const float *bottom, const float *top,
                                               int_tp index,
                                               int_tp numImages, kernelConfig* config)
{
    // warm up.
    bool saved_tuned = tuned_;
    tuned_ = false;
    convolve(bottom, top, index, num_, config);
    Timer timer;
    timer.initted();
    timer.Start();
    cl_int err;
    dbgPrint(std::cout << "Bechmarking kernel: " << config->kernelName << std::endl);
    tuned_ = true;
    int loop_cnt = 4;
    for (int i = 0; i < loop_cnt; i++) {
        err = convolve(bottom, top, index, num_, config);
        if (err != CL_SUCCESS)
            break;
    }
    tuned_ = saved_tuned;
    timer.Stop();
    if (err != CL_SUCCESS) {
        config->tested = true;
        config->verified = false;
        dbgPrint(std::cout << "convolution failed with error code " << err << std::endl);
        return 1e5;
    }

    float elapsedTime = timer.MilliSeconds() / loop_cnt;
    #ifdef dbg
    double out_w = output_w_;
    double out_h = output_h_;
    double out_z = M_;
    double k_w = kernel_w_;
    double k_h = kernel_h_;
    double k_z = channels_;
    double totalFlops = ((k_w*k_h*k_z -1)*2)*(out_w*out_h*out_z)*num_;
    std::cout << "\tEstimated Gflops:" << ((totalFlops/1000)/1000)/1000
              << std::endl;
    std::cout << "\tEstimated GFLOPS/S: " << (((totalFlops/1000)/1000)/1000)*(1000.0/elapsedTime)
              << std::endl;
    #if 0
    std::cout << "Estimated utilization: " <<
        ((((totalFlops/1000)/1000)/1000)*(1000.0/elapsedTime))/880.0
        << std::endl;
    #endif
    #endif
    return elapsedTime;
}

template<>
bool LibDNNConvSpatial<float>::verify_result(const float *bottom, float *top,
                                             int_tp index,
                                             int_tp numImages, const float *verify_blob, kernelConfig* config)
{

    uint_tp verificationFail = 0;

    if (config->verified)
        return true;
    else if (config->tested)
        return false;

    greentea_gpu_set(0, numImages * this->top_dim_, (float) 0, (cl_mem) top, 0);
    config->executionTime = timed_convolve(bottom, top, index, numImages, config);
    const float *verify_data;
    float *data;
    float *tmp_verify_data;
    ocl::Queue queue = ocl::Queue::getDefault();
    data = reinterpret_cast<float *>(clEnqueueMapBuffer((cl_command_queue)queue.ptr(),
                                                        (cl_mem)top, true, CL_MAP_READ,
                                                        0, sizeof(float) * numImages * this->top_dim_,
                                                        0, NULL, NULL, NULL));
    tmp_verify_data =
        reinterpret_cast<float *>(clEnqueueMapBuffer((cl_command_queue)queue.ptr(),
                                                     (cl_mem)verify_blob, true, CL_MAP_READ,
                                                     0, sizeof(float) * numImages * this->top_dim_,
                                                     0, NULL, NULL, NULL));
    verify_data = tmp_verify_data;

    for (int_tp n = 0; n < numImages; ++n) {
        for (int_tp g = 0; g < group_; ++g) {
            int_tp output_image_offset = n * this->top_dim_ + output_w_ * output_h_ * M_ * g;
            for (int out_ch = 0; out_ch < M_ && !verificationFail; out_ch++)
                for (int h = 0; h < output_h_ && !verificationFail; h++)
                    for (int w = 0; w < output_w_; w++) {
                        size_t offset = output_image_offset + out_ch * output_w_ * output_h_ + h * output_w_ + w;
                        if (fabs(data[offset] - verify_data[offset]) > 0.1 * fabs(verify_data[offset]) &&
                            !(fabs(verify_data[offset]) < 1.e-3 &&
                            fabs(data[offset] - verify_data[offset]) < 1.e-4))
                        {
                            dbgPrint(printf("test verification failed @ image %d group %d"
                                            "out_ch %d h %d w %d got %G expected %G\n",
                                            n, g, out_ch, h, w, data[offset], verify_data[offset]));
                            verificationFail = 1;
                            goto out;
                        }
                    }
        }
    }

out:
    clEnqueueUnmapMemObject((cl_command_queue)queue.ptr(),
                            (cl_mem)top, data, 0, NULL, NULL);
    clEnqueueUnmapMemObject((cl_command_queue)queue.ptr(),
                            (cl_mem)verify_blob, tmp_verify_data, 0, NULL, NULL);
    if (verificationFail == 1)
        return false;
    else
        return true;
}

template<typename Dtype>
ocl::Program LibDNNConvSpatial<Dtype>::compile_fw_kernel()
{
    String errmsg;
    ocl::Context ctx = ocl::Context::getDefault();
    ocl::ProgramSource src(kernel_.c_str());
    ocl::Program program = ctx.getProg(src, options_, errmsg);
    if (!kernel_name_.empty())
        phash.insert(std::pair<std::string, ocl::Program>(kernel_name_, program));
    return program;
}

template<>
bool LibDNNConvSpatial<float>::create_gemm_like_conv_kernel(const float *bottom, const float *top,
                                                            int_tp blockM,
                                                            int_tp blockK, int_tp blockN)
{

    int_tp workItemOutput[3] = { blockM, blockK, blockN };
    int_tp output_width = output_w_;
    int_tp output_height = output_h_;
    int_tp simd_size = blockK;
    int_tp num_batches = num_;
    int_tp alignedFilterWidth = ALIGN(M_, blockN);
    int_tp alignedExpandHeight = ALIGN(output_width * output_height, blockM);
    int_tp globalWorkSizeDX = blockN;
    int_tp globalWorkSizeDY = blockM;
    size_t sgemm_m = alignedExpandHeight;
    size_t sgemm_n = alignedFilterWidth;
    size_t gx = (size_t) ceil( (float) sgemm_n / (float) globalWorkSizeDX );  // NOLINT
    size_t gy = (size_t) ceil( (float) sgemm_m / (float) globalWorkSizeDY );  // NOLINT
    gy = ALIGN(gy, blockK);
    size_t gz = num_batches;
    size_t global_size[3] = { gx, gy, gz };
    size_t local_size[3] = { 1, static_cast<size_t>(simd_size), 1 };

    kernelType_ = 5;
    blockM_ = blockM;
    blockK_ = blockK;
    blockN_ = blockN;
    GenerateKernels();
    ocl::Program program = compile_fw_kernel();

    size_t workgroupSize_used;
    ocl::Kernel kernel(kernel_name_.c_str(), program);
    cl_int err = clGetKernelWorkGroupInfo((cl_kernel)kernel.ptr(),
                                          (cl_device_id)ocl::Device::getDefault().ptr(),
                                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                          sizeof(size_t), &workgroupSize_used,
                                          NULL);

    if (workgroupSize_used != simd_size) {
        phash.erase(kernel_name_);
        return false;
    }

    if (err == CL_SUCCESS || err == true) {
        kernelQueue.push_back(new kernelConfig(kernel_name_, global_size, local_size, workItemOutput,
                                               false, true, false, 5));
        return true;
    } else {
        phash.erase(kernel_name_);
        return false;
    }
}

template<>
bool LibDNNConvSpatial<float>::setup_IDLF(const float *bottom, const float *top,
                                          int_tp blockWidth,
                                          int_tp blockHeight, int_tp simd_size)
{
    int_tp workItemOutput[3] = { blockWidth, blockHeight, simd_size };
    const int_tp num_output_maps = M_;
    int_tp output_width = output_w_;
    int_tp output_height = output_h_;
    int_tp output_block_width = blockWidth;
    int_tp output_block_height = blockHeight;
    int_tp num_batches = num_;

    size_t global_size[3] = {
        (size_t) (output_width + output_block_width - 1) / output_block_width,
        (size_t) (output_height + output_block_height - 1) / output_block_height,
        (size_t) num_batches * ALIGN(num_output_maps, simd_size) };
    size_t local_size[3] = { 1, 1, static_cast<size_t>(simd_size) };

    kernelType_ = KERNEL_TYPE_INTEL_IDLF;
    blockM_ = blockWidth;
    blockK_ = blockHeight;
    blockN_ = simd_size;

    GenerateKernels();
    ocl::Program program = compile_fw_kernel();

    // ClKernel kernel;
    size_t workgroupSize_used;
    ocl::Kernel kernel(kernel_name_.c_str(), program);
    cl_int err = clGetKernelWorkGroupInfo((cl_kernel)kernel.ptr(),
                                          (cl_device_id)ocl::Device::getDefault().ptr(),
                                          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                          sizeof(size_t), &workgroupSize_used,
                                          NULL);

    if (workgroupSize_used != simd_size) {
        phash.erase(kernel_name_);
        return false;
    }

    if (err == CL_SUCCESS || err == true) {
        kernelQueue.push_back(new kernelConfig(kernel_name_, global_size, local_size, workItemOutput,
                                               false, true, false, 2));
        return true;
    } else {
        phash.erase(kernel_name_);
        return false;
    }
}

template<>
bool LibDNNConvSpatial<float>::tune_local_size(const float *bottom, const float *top,
                                               kernelConfig* config)
{
    if (config->use_null_local || !config->autoTune)
        return true;

    float fastestTime = 999999990000000000000000000.0f;
    uint_tp multiplier = 4;
    uint_tp localSize[3] = { 1, 1, 1 };

    int_tp skip = 0;
    Timer timer;
    timer.initted();
    bool allFailed = true;
    for (int_tp z = 0; z <= 16; z++) {
        for (int_tp y = 0; y <= 16; y++) {
            for (int_tp x = 1; x <= 16; x++) {
                timer.Start();
                skip = 0;

                if (config->autoTune) {
                    config->local_work_size[0] =
                        (multiplier * x == 0) ? 1 : multiplier * x;
                    config->local_work_size[1] =
                        (multiplier * y == 0) ? 1 : multiplier * y;
                    config->local_work_size[2] =
                        (multiplier * z == 0) ? 1 : multiplier * z;

                    calculate_global_size(1, config->workItem_output,
                                          config->local_work_size,
                                          config->global_work_size);
                }
                if (config->workItem_output[2] * config->global_work_size[2] != M_)
                    break;

                if (config->swizzle_weights)
                    z = 32;

                int_tp err = 0;
                err = convolve(bottom, top, 0, 1, config);

                if (err != CL_SUCCESS)
                    skip = 1;

                if (skip) {
                    timer.Stop();
                    break;
                }
                timer.Stop();
                allFailed = false;
                float elapsedTime = timer.MilliSeconds();

                if (elapsedTime < fastestTime) {
                    fastestTime = elapsedTime;
                    localSize[0] = config->local_work_size[0];
                    localSize[1] = config->local_work_size[1];
                    localSize[2] = config->local_work_size[2];
                }
            }
        }
    }
    if (allFailed) {
        // 1,1,1 is never a good local size and no need to test at all.
        dbgPrint(std::cout << "Can't find good local size for " << config->kernelName << std::endl);
        return false;
    }

    dbgPrint(std::cout << "Best local size[" << localSize[0] << "]["
             << localSize[1] << "]["<< localSize[2] << "]: " << fastestTime
             << " Kernel_h: " << kernel_h_ << " kernel_w_: " << kernel_w_
             << " stride_w: " << stride_w_ << " pad_w_: " << pad_w_ << std::endl);

    if (config->autoTune) {
        for (int_tp li = 0; li < 3; li++)
            config->local_work_size[li] = localSize[li];

        calculate_global_size(1, config->workItem_output, config->local_work_size,
                              config->global_work_size);
    }
    return true;
}

template<>
void LibDNNConvSpatial<float>::create_convolution_kernel(const float *bottom, const float *top,
                                                         int_tp kernelType,
                                                         int_tp blockWidth, int_tp blockHeight,
                                                         int_tp blockDepth)
{
    if (kernelType == 2)
        setup_IDLF(bottom, top, blockWidth, blockHeight, blockDepth);
    else if (kernelType == 4)
        create_basic_kernel(bottom, top, blockWidth, blockHeight, blockDepth);
    else if (kernelType == 5)
        create_gemm_like_conv_kernel(bottom, top, blockWidth, blockHeight, blockDepth);
    else
        assert(0);
}

template<>
void LibDNNConvSpatial<float>::setup_convolution(const float *bottom, float *top,
                                                 const float *verify_blob)
{

    if (ocl::Device::getDefault().intelSubgroupsSupport()) {
        /* IDLF kernels are using Intel specific extension which make
           them intel only. */
        // Generates static key_
        int max_compute_units = ocl::Device::getDefault().maxComputeUnits();
        int kernelCnt = 0;
        if (this->group_ == 1 && ((M_ % 8 == 0) && (M_ % 32 != 24))) {
            create_convolution_kernel(bottom, top, 5, 1, 8, 32);
            create_convolution_kernel(bottom, top, 5, 2, 8, 32);
            if (kernel_w_ < 4 && M_ % 32 == 0)
                create_convolution_kernel(bottom, top, 5, 1, 16, 32);
        }

        for (int simd_size = 8; simd_size <= 16; simd_size += 8) {
            if (simd_size == 8 && !((this->group_ == 1 || M_ % 8 == 0)))
                continue;
            if (simd_size == 16 && !(this->group_ == 1 || M_ % 16 == 0))
                continue;
            int width_max, height_max, block_size_max;
            if (simd_size == 8) {
                width_max = 16;
                height_max = 16;
                block_size_max = 64;
            } else {
                width_max = 14;
                height_max = 14;
                block_size_max = 32;
            }
            for (uint32_t width = width_max; width > 0; width--) {
                int candidate = 0;
                if (width > output_w_)
                    continue;
                for (uint32_t height = height_max; height > 0; height--) {
                    if (width * height > block_size_max || height > output_h_)
                        continue;
                    // Only when the work items count is less than the device
                    // max work items or the M_ is less than 16, we will tune
                    // for simd 8.
                    if (simd_size == 8 &&
                        M_ >= 16 &&
                        ((num_ * M_ * output_w_ * output_h_ / static_cast<float>(width * height)) >=
                        max_compute_units * 7 * 16))
                        continue;
                    int tile_x = (kernel_w_ * dilation_w_ + (width - 1) * stride_w_ + 3) & ~3;
                    int tile_y = kernel_h_ * dilation_h_ + (height - 1) * stride_h_;
                    if (tile_x > (4 * simd_size))
                        continue;
                    int tile_y_stride = (4 * simd_size) / tile_x;

                    if ((tile_y + tile_y_stride - 1) / tile_y_stride < 4) {
                        create_convolution_kernel(bottom, top, 2, width, height, simd_size);
                        candidate++;
                    }
                    if (candidate >= 4 && height == 2)
                        break;
                }
                kernelCnt += candidate;
                if (kernelCnt >= 12 && width == 2)
                    break;
            }
        }
    }
    for (int_tp x = 0; x < kernelQueue.size(); x++) {
        if (tune_local_size(bottom, top, kernelQueue[x])) {
            kernelQueue[x]->executionTime = timed_convolve(bottom, top, bottom_index_,
                                                           num_, kernelQueue[x]);
        } else {
            // skip those kernels without a good local size.
            kernelQueue[x]->verified = false;
            kernelQueue[x]->tested = true;
        }
        #ifdef TEST_ALL_KERNELS
        if (kernelQueue[x]->tested == false) {
            bool verified = verify_result(bottom, top, bottom_index_, num_,
                                          verify_blob, kernelQueue[x]);
            if (verified == false) {
                dbgPrint(std::cout << "Kernel "
                         << kernelQueue[x]->kernelName
                         << " failed verification" << std::endl);
                dbgPrint(std::cout << "kernelQueue[x]->workItem_output[0]: "
                         << kernelQueue[x]->workItem_output[0] << " "
                         << "kernelQueue[x]->workItem_output[1]: "
                         << kernelQueue[x]->workItem_output[1] << " "
                         << "kernelQueue[x]->workItem_output[2]: "
                         << kernelQueue[x]->workItem_output[2] << " "
                         << "kernelQueue[x]->kernelType: "
                         << kernelQueue[x]->kernelType << " "
                         << "kernelQueue[x]->global_work_size[0]: "
                         << kernelQueue[x]->global_work_size[0] << " "
                         << "kernelQueue[x]->global_work_size[1]: "
                         << kernelQueue[x]->global_work_size[1] << " "
                         << "kernelQueue[x]->global_work_size[2]: "
                         << kernelQueue[x]->global_work_size[2] << " "
                         << "kernelQueue[x]->local_work_size[0]: "
                         << kernelQueue[x]->local_work_size[0] << " "
                         << "kernelQueue[x]->local_work_size[1]: "
                         << kernelQueue[x]->local_work_size[1] << " "
                         << "kernelQueue[x]->local_work_size[2]: "
                         << kernelQueue[x]->local_work_size[2] << " "
                         << kernelQueue[x]->swizzle_weights << " "
                         << kernelQueue[x]->use_null_local << std::endl);
            } else {
                dbgPrint(std::cout << "Kernel "
                         << kernelQueue[x]->kernelName
                         << " pass verification" << std::endl);
            }
        }
        #endif
    }
    int_tp failures = 0;
    bool verification = false;
    if (kernelQueue.size()) {
        while (failures < kernelQueue.size()) {
            int_tp fastestKernel = -1;
            float fastestTime = 999999990000000000000000000.0f;

            for (int_tp x = 0; x < kernelQueue.size(); x++) {
                if (kernelQueue[x]->executionTime < fastestTime &&
                    kernelQueue[x]->tested == false) {
                    fastestKernel = x;
                    fastestTime = kernelQueue[x]->executionTime;
                }
            }
            if (fastestKernel < 0) break;
            // Test fastest kernel
            bool verified = verify_result(bottom, top, bottom_index_, num_,
                                          verify_blob, kernelQueue[fastestKernel]);
            if (verified == true) {
                kernelQueue[fastestKernel]->verified = true;
                kernel_index_ = fastestKernel;
                verification = true;
                break;
            } else {
                kernelQueue[fastestKernel]->tested = true;
                dbgPrint(std::cout << "Kernel " <<
                         kernelQueue[fastestKernel]->kernelName <<
                         " failed verification" << std::endl);
                failures++;
            }
        }
    }
    if (verification) {
        dbgPrint(std::cout << "Kernel <" << kernelQueue[kernel_index_]->kernelName <<
                 "> passed verification" << std::endl);
    } else {
        dbgPrint(std::cout << "Verification was not successful, " <<
                 "fallback to basic kernel" << std::endl);
        create_basic_kernel(bottom, top, 1, 1, 1);
        kernel_index_ = kernelQueue.size() - 1;
        verification = verify_result(bottom, top, bottom_index_, num_,
                                     verify_blob, kernelQueue[kernel_index_]);
        CHECK_EQ(verification, true) << "Basic kernel failed verification." << std::endl;
    }
    this->bestKernelConfig = kernelQueue[kernel_index_];

    dbgPrint(std::cout << "Convolution Time:" << kernelQueue[kernel_index_]->executionTime << std::endl);

    if (bestKernelConfig->kernelType != 2 && bestKernelConfig->kernelType != 5)
        swizzled_weights_ = NULL;

    for (int_tp x = 0; x < kernelQueue.size(); x++) {
        if (x != kernel_index_) {
            phash.erase(kernelQueue[x]->kernelName);
            delete kernelQueue[x];
        }
    }
    kernelQueue.clear();

    tuned_ = true;

    std::string outputFile;
    outputFile = cache_path_.str() + key_;
    std::ifstream cachedKernel(outputFile.c_str());
    std::ofstream outputKernel;
    outputKernel.open(outputFile.c_str());
    outputKernel << bestKernelConfig->workItem_output[0] << " "
                 << bestKernelConfig->workItem_output[1] << " "
                 << bestKernelConfig->workItem_output[2] << " "
                 << bestKernelConfig->kernelType << " "
                 << bestKernelConfig->global_work_size[0] << " "
                 << bestKernelConfig->global_work_size[1] << " "
                 << bestKernelConfig->global_work_size[2] << " "
                 << bestKernelConfig->local_work_size[0] << " "
                 << bestKernelConfig->local_work_size[1] << " "
                 << bestKernelConfig->local_work_size[2] << " "
                 << bestKernelConfig->swizzle_weights << " "
                 << 0 << " "  // deprecated
                 << bestKernelConfig->use_null_local << " ";
    outputKernel.close();
}

template<typename Dtype>
void LibDNNConvSpatial<Dtype>::load_cached_kernels(const Dtype *bottom, const Dtype *top)
{
    // Generates static key_
    std::string previous_key = key_;
    generate_key();
    int prev_kernel_type = 0;
    if (tuned_) {
        if (key_.compare(previous_key) == 0)
            return;
        tuned_ = false;
        prev_kernel_type = bestKernelConfig->kernelType;
        phash.erase(bestKernelConfig->kernelName);
        delete bestKernelConfig;
        bestKernelConfig = NULL;
    }

    // Find cached kernel configuration
    std::string outputFile;
    outputFile = cache_path_.str() + key_;
    std::ifstream cachedKernel(outputFile.c_str());
    if (cachedKernel) {
        int_tp x, y, z, type;
        cachedKernel >> x;
        cachedKernel >> y;
        cachedKernel >> z;
        cachedKernel >> type;
        if (type == 2) {
            if (z == 1)
                z = 16;
            CHECK_EQ(z == 16 || z == 8, true) << "invalid SIMD size" << std::endl;
        }
        create_convolution_kernel(bottom, top, type, x, y, z);
        kernel_index_ = kernelQueue.size() - 1;
        if (kernel_index_ == -1) {
            std::cerr << "Failed to get kernel from cached configurations."
                      << std::endl;
            std::cerr << "Deleting broken cache file and try tuning again..."
                      << std::endl;
            std::string bakFile = outputFile + ".bak";
            std::rename(outputFile.c_str(), bakFile.c_str());
            return;
        }
        bestKernelConfig = kernelQueue[kernel_index_];
        kernelQueue.clear();
        // As we are using varying image size kernels now, let's skip the
        // cached work group size and local group size here, and we already
        // get correct work/local group size at the create_convolution kernel stage.
        // To not break the previous trained record, for now just skipping them.
        // Will use a totally different cache mechanism in the future.
        size_t foo;  // for deprecated parameters.
        cachedKernel >> foo;
        cachedKernel >> foo;
        cachedKernel >> foo;
        cachedKernel >> bestKernelConfig->local_work_size[0];
        cachedKernel >> bestKernelConfig->local_work_size[1];
        cachedKernel >> bestKernelConfig->local_work_size[2];
        if (bestKernelConfig->kernelType == 1)
            calculate_global_size(1, bestKernelConfig->workItem_output,
                                  bestKernelConfig->local_work_size,
                                  bestKernelConfig->global_work_size);
        cachedKernel >> bestKernelConfig->swizzle_weights;
        cachedKernel >> foo;
        cachedKernel >> bestKernelConfig->use_null_local;
        tuned_ = true;
        // If kernel type changed to type 2 or 4, we need to reset the swizzled
        // weights pointer to invalidate the previous swizzled weights data.
        if (prev_kernel_type != bestKernelConfig->kernelType &&
            (bestKernelConfig->kernelType == 2 || bestKernelConfig->kernelType == 5))
            swizzled_weights_ = NULL;
    }
    return;
}

template<typename Dtype>
void LibDNNConvSpatial<Dtype>::SetUp(const Dtype *bottom, const Dtype *top)
{
    {
        load_cached_kernels(bottom, top);
    }
}

template void LibDNNConvSpatial<float>::SetUp(const float *bottom, const float *top);

template void LibDNNConvSpatial<double>::SetUp(const double *bottom, const double *top);

template void LibDNNConvSpatial<float>::swizzleWeights(const float *bottom,
                                                       const float *top,
                                                       int_tp swizzle_factor,
                                                       bool interleave = false);
template void LibDNNConvSpatial<double>::swizzleWeights(const double *bottom,
                                                        const double *top,
                                                        int_tp swizzle_factor,
                                                        bool interleave = false);

template<>
void LibDNNConvSpatial<double>::create_convolution_kernel(const double *bottom, const double *top,
                                                          int_tp kernelType,
                                                          int_tp blockWidth, int_tp blockHeight,
                                                          int_tp blockDepth)
{
    NOT_IMPLEMENTED;
    return;
}

template<>
bool LibDNNConvSpatial<double>::setup_IDLF(const double *bottom, const double *top,
                                           int_tp blockWidth,
                                           int_tp blockHeight, int_tp blockDepth)
{
    NOT_IMPLEMENTED;
    return false;
}

template<>
bool LibDNNConvSpatial<double>::create_gemm_like_conv_kernel(const double *bottom, const double *top,
                                                             int_tp blockWidth,
                                                             int_tp blockHeight, int_tp blockDepth)
{
    NOT_IMPLEMENTED;
    return false;
}


template<>
bool LibDNNConvSpatial<double>::verify_result(const double *bottom, double *top,
                                              int_tp index,
                                              int_tp numImages, const double *verify_blob, kernelConfig* config)
{
    NOT_IMPLEMENTED;
    return false;
}

template<>
bool LibDNNConvSpatial<double>::create_basic_kernel(const double *bottom, const double *top,
                                                    int_tp blockWidth,
                                                    int_tp blockHeight, int_tp blockDepth)
{
    NOT_IMPLEMENTED;
    return false;
}

template<>
bool LibDNNConvSpatial<double>::tune_local_size(const double *bottom, const double *top,
                                                kernelConfig* config)
{
    NOT_IMPLEMENTED;
    return false;
}

template<>
cl_int LibDNNConvSpatial<double>::convolve(const double *bottom, const double *top,
                                           int_tp index,
                                           int_tp numImages, kernelConfig* config)
{
    NOT_IMPLEMENTED;
    return false;
}

template<>
float LibDNNConvSpatial<double>::timed_convolve(const double *bottom, const double *top,
                                                int_tp index,
                                                int_tp numImages, kernelConfig* config)
{
    NOT_IMPLEMENTED;
    return 0.f;
}

template<>
void LibDNNConvSpatial<double>::setup_convolution(const double *bottom, double *top,
                                                  const double *verify_blob)
{
    NOT_IMPLEMENTED;
}

template<>
void LibDNNConvSpatial<double>::calculate_global_size(int_tp batch,
                                                      int_tp* workItemOutput,
                                                      size_t* localSizes, size_t* globalSizes)
{
    NOT_IMPLEMENTED;
}

//INSTANTIATE_CLASS(LibDNNConvSpatial);
template class LibDNNConvSpatial<float>;
template class LibDNNConvSpatial<double>;
#endif // HAVE_OPENCL

}  // namespace caffe
