#include <errno.h>
#include <string.h>
#include <stdlib.h>

#include <mruby.h>
#include <mruby/proc.h>
#include <mruby/data.h>
#include <mruby/numeric.h>
#include <mruby/string.h>
#include <mruby/array.h>
#include <mruby/class.h>
#include <mruby/variable.h>
#include <tensorflow/lite/c/c_api.h>

#if 1
#define ARENA_SAVE \
  int ai = mrb_gc_arena_save(mrb);
#define ARENA_RESTORE \
  mrb_gc_arena_restore(mrb, ai);
#else
#define ARENA_SAVE
#define ARENA_RESTORE
#endif

static const char*
tensor_type_name(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return "none";
    case kTfLiteFloat32:
      return "float32";
    case kTfLiteInt32:
      return "int32";
    case kTfLiteUInt8:
      return "uint8";
    case kTfLiteInt64:
      return "int64";
    case kTfLiteString:
      return "string";
    case kTfLiteBool:
      return "bool";
    case kTfLiteInt16:
      return "int16";
    case kTfLiteComplex64:
      return "complex64";
    case kTfLiteInt8:
      return "int8";
    default:
      return "unknown";
  }
}

static void
mrb_tflite_model_free(mrb_state *mrb, void *p) {
  TfLiteModelDelete((TfLiteModel*)p);
}

static void
mrb_tflite_interpreter_options_free(mrb_state *mrb, void *p) {
  TfLiteInterpreterOptionsDelete((TfLiteInterpreterOptions*)p);
}

static void
mrb_tflite_interpreter_free(mrb_state *mrb, void *p) {
  TfLiteInterpreterDelete((TfLiteInterpreter*)p);
}

static const struct mrb_data_type mrb_tflite_tensor_type_ = {
  "mrb_tflite_tensor", NULL,
};

static const struct mrb_data_type mrb_tflite_model_type = {
  "mrb_tflite_model", mrb_tflite_model_free,
};

static const struct mrb_data_type mrb_tflite_interpreter_options_type = {
  "mrb_tflite_interpreter_options", mrb_tflite_interpreter_options_free
};

static const struct mrb_data_type mrb_tflite_interpreter_type = {
  "mrb_tflite_interpreter", mrb_tflite_interpreter_free
};

static mrb_value
mrb_tflite_model_init(mrb_state *mrb, mrb_value self) {
  TfLiteModel* model;
  mrb_value str;
  mrb_get_args(mrb, "S", &str);
  model = TfLiteModelCreate(RSTRING_PTR(str), RSTRING_LEN(str));
  if (model == NULL) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "cannot create model");
  }
  DATA_TYPE(self) = &mrb_tflite_model_type;
  DATA_PTR(self) = model;
  return self;
}

static mrb_value
mrb_tflite_model_from_file(mrb_state *mrb, mrb_value self) {
  TfLiteModel* model;
  mrb_value str;
  struct RClass* _class_tflite_model;

  mrb_get_args(mrb, "S", &str);
  model = TfLiteModelCreateFromFile(RSTRING_PTR(str));
  if (model == NULL) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "cannot create model");
  }
  _class_tflite_model = mrb_class_get_under(mrb, mrb_module_get(mrb, "TfLite"), "Model");
  return mrb_obj_value(Data_Wrap_Struct(mrb, (struct RClass*) _class_tflite_model,
    &mrb_tflite_model_type, (void*) model));
}

static mrb_value
mrb_tflite_interpreter_options_init(mrb_state *mrb, mrb_value self) {
  TfLiteInterpreterOptions* interpreter_options;

  interpreter_options = TfLiteInterpreterOptionsCreate();
  if (interpreter_options == NULL) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "cannot create interpreter options");
  }
  DATA_TYPE(self) = &mrb_tflite_interpreter_options_type;
  DATA_PTR(self) = interpreter_options;
  return self;
}

static mrb_value
mrb_tflite_interpreter_options_num_threads_set(mrb_state *mrb, mrb_value self) {
  TfLiteInterpreterOptions* interpreter_options;
  int num_threads = 0;
  mrb_get_args(mrb, "i", &num_threads);
  interpreter_options = DATA_PTR(self);
  TfLiteInterpreterOptionsSetNumThreads(interpreter_options, num_threads);
  return mrb_nil_value();
}

static mrb_value
mrb_tflite_interpreter_options_add_delegate(mrb_state *mrb, mrb_value self) {
  TfLiteInterpreterOptions* interpreter_options;
  mrb_value delegate;
  mrb_get_args(mrb, "o", &delegate);
  interpreter_options = DATA_PTR(self);
  TfLiteInterpreterOptionsAddDelegate(interpreter_options, DATA_PTR(delegate));
  return mrb_nil_value();
}

static mrb_value
mrb_tflite_interpreter_init(mrb_state *mrb, mrb_value self) {
  TfLiteInterpreter* interpreter;
  TfLiteInterpreterOptions* interpreter_options = NULL;
  mrb_value arg_model;
  mrb_value arg_options = mrb_nil_value();

  mrb_get_args(mrb, "o|o", &arg_model, &arg_options);
  if (mrb_nil_p(arg_model) || DATA_TYPE(arg_model) != &mrb_tflite_model_type) {
    mrb_raise(mrb, E_ARGUMENT_ERROR, "invalid argument");
  }
  if (!mrb_nil_p(arg_options) && DATA_TYPE(arg_options) == &mrb_tflite_interpreter_options_type) {
    interpreter_options = DATA_PTR(arg_options);
  }
  interpreter = TfLiteInterpreterCreate((TfLiteModel*) DATA_PTR(arg_model), interpreter_options);
  if (interpreter == NULL) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "cannot create interpreter");
  }
  DATA_TYPE(self) = &mrb_tflite_interpreter_type;
  DATA_PTR(self) = interpreter;
  return self;
}

static mrb_value
mrb_tflite_interpreter_allocate_tensors(mrb_state *mrb, mrb_value self) {
  TfLiteInterpreter* interpreter = DATA_PTR(self);
  if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "cannot allocate tensors");
  }
  return mrb_nil_value();
}

static mrb_value
mrb_tflite_interpreter_invoke(mrb_state *mrb, mrb_value self) {
  TfLiteInterpreter* interpreter = DATA_PTR(self);
  if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
    mrb_raise(mrb, E_RUNTIME_ERROR, "cannot invoke");
  }
  return mrb_nil_value();
}

static mrb_value
mrb_tflite_interpreter_input_tensor_count(mrb_state *mrb, mrb_value self) {
  TfLiteInterpreter* interpreter = DATA_PTR(self);
  return mrb_fixnum_value(TfLiteInterpreterGetInputTensorCount(interpreter));
}

static mrb_value
mrb_tflite_interpreter_output_tensor_count(mrb_state *mrb, mrb_value self) {
  TfLiteInterpreter* interpreter = DATA_PTR(self);
  return mrb_fixnum_value(TfLiteInterpreterGetOutputTensorCount(interpreter));
}

static mrb_value
mrb_tflite_interpreter_input_tensor(mrb_state *mrb, mrb_value self) {
  TfLiteTensor* tensor;
  mrb_int index;
  struct RClass* _class_tflite_tensor;
  mrb_value c;
  TfLiteInterpreter* interpreter = DATA_PTR(self);
  mrb_get_args(mrb, "i", &index);
  tensor = TfLiteInterpreterGetInputTensor(interpreter, index);
  if (tensor == NULL) {
    mrb_raise(mrb, E_ARGUMENT_ERROR, "invalid argument");
  }
  _class_tflite_tensor = mrb_class_get_under(mrb, mrb_module_get(mrb, "TfLite"), "Tensor");
  c = mrb_obj_new(mrb, _class_tflite_tensor, 0, NULL);
  DATA_TYPE(c) = &mrb_tflite_tensor_type_;
  DATA_PTR(c) = tensor;
  return c;
}

static mrb_value
mrb_tflite_interpreter_output_tensor(mrb_state *mrb, mrb_value self) {
  TfLiteTensor* tensor;
  mrb_int index;
  struct RClass* _class_tflite_tensor;
  mrb_value c;
  TfLiteInterpreter* interpreter = DATA_PTR(self);
  mrb_get_args(mrb, "i", &index);
  tensor = (TfLiteTensor*) TfLiteInterpreterGetOutputTensor(interpreter, index);
  _class_tflite_tensor = mrb_class_get_under(mrb, mrb_module_get(mrb, "TfLite"), "Tensor");
  c = mrb_obj_new(mrb, _class_tflite_tensor, 0, NULL);
  DATA_TYPE(c) = &mrb_tflite_tensor_type_;
  DATA_PTR(c) = tensor;
  return c;
}

static mrb_value
mrb_tflite_tensor_type(mrb_state *mrb, mrb_value self) {
  TfLiteTensor* tensor = DATA_PTR(self);
  return mrb_fixnum_value(TfLiteTensorType(tensor));
}

static mrb_value
mrb_tflite_tensor_name(mrb_state *mrb, mrb_value self) {
  TfLiteTensor* tensor = DATA_PTR(self);
  return mrb_str_new_cstr(mrb, TfLiteTensorName(tensor));
}

static mrb_value
mrb_tflite_tensor_num_dims(mrb_state *mrb, mrb_value self) {
  TfLiteTensor* tensor = DATA_PTR(self);
  return mrb_fixnum_value(TfLiteTensorNumDims(tensor));
}

static mrb_value
mrb_tflite_tensor_dim(mrb_state *mrb, mrb_value self) {
  TfLiteTensor* tensor = DATA_PTR(self);
  mrb_int index;
  mrb_get_args(mrb, "i", &index);
  return mrb_fixnum_value(TfLiteTensorDim(tensor, index));
}

static mrb_value
mrb_tflite_tensor_byte_size(mrb_state *mrb, mrb_value self) {
  TfLiteTensor* tensor = DATA_PTR(self);
  return mrb_fixnum_value(TfLiteTensorByteSize(tensor));
}

static mrb_value
mrb_tflite_tensor_data_get(mrb_state *mrb, mrb_value self) {
  TfLiteTensor* tensor = DATA_PTR(self);
  int ai, i;
  mrb_value ret;
  int len;
  TfLiteType type;
  uint8_t *uint8s;
  float *float32s;

  type = TfLiteTensorType(tensor);
  switch (type) {
    case kTfLiteUInt8:
    case kTfLiteInt8:
      len = TfLiteTensorByteSize(tensor);
      uint8s = (uint8_t*) TfLiteTensorData(tensor);
      ret = mrb_ary_new_capa(mrb, len);
      ai = mrb_gc_arena_save(mrb);
      for (i = 0; i < len; i++) {
        mrb_ary_push(mrb, ret, mrb_fixnum_value(uint8s[i]));
        mrb_gc_arena_restore(mrb, ai);
      }
      break;
    case kTfLiteFloat32:
      len = TfLiteTensorByteSize(tensor) / 4;
      float32s = (float*) TfLiteTensorData(tensor);
      ret = mrb_ary_new_capa(mrb, len);
      ai = mrb_gc_arena_save(mrb);
      for (i = 0; i < len; i++) {
        mrb_ary_push(mrb, ret, mrb_float_value(mrb, float32s[i]));
        mrb_gc_arena_restore(mrb, ai);
      }
      break;
    default:
      mrb_raisef(mrb, E_RUNTIME_ERROR, "tensor type %S not supported", mrb_str_new_cstr(mrb, tensor_type_name(type)));
  }
  MRB_SET_FROZEN_FLAG(mrb_basic_ptr(ret));
  return ret;
}

static mrb_value
mrb_tflite_tensor_data_set(mrb_state *mrb, mrb_value self) {
  TfLiteTensor* tensor = DATA_PTR(self);
  int i;
  int len, ary_len;
  TfLiteType type;
  uint8_t *uint8s;
  float *float32s;
  mrb_value arg_data;

  mrb_get_args(mrb, "o", &arg_data);
  if (mrb_nil_p(arg_data) || mrb_type(arg_data) != MRB_TT_ARRAY) {
    mrb_raise(mrb, E_ARGUMENT_ERROR, "argument must be array");
  }
  ary_len = RARRAY_LEN(arg_data);

  type = TfLiteTensorType(tensor);
  switch (type) {
    case kTfLiteUInt8:
    case kTfLiteInt8:
      len = TfLiteTensorByteSize(tensor);
      if (ary_len != len) {
        mrb_raise(mrb, E_ARGUMENT_ERROR, "argument size mismatched");
      }
      uint8s = (uint8_t*) TfLiteTensorData(tensor);
      for (i = 0; i < len; i++) {
        uint8s[i] = (uint8_t) mrb_fixnum(mrb_ary_entry(arg_data, i));
      }
      break;
    case kTfLiteFloat32:
      len = TfLiteTensorByteSize(tensor) / 4;
      if (ary_len != len) {
        mrb_raise(mrb, E_ARGUMENT_ERROR, "argument size mismatched");
      }
      float32s = (float*) TfLiteTensorData(tensor);
      for (i = 0; i < len; i++) {
        float32s[i] = mrb_float(mrb_ary_entry(arg_data, i));
      }
      break;
    default:
      mrb_raisef(mrb, E_RUNTIME_ERROR, "tensor type %S not supported", mrb_str_new_cstr(mrb, tensor_type_name(type)));
  }
  return mrb_nil_value();
}

void
mrb_mruby_tflite_gem_init(mrb_state* mrb) {
  struct RClass *_class_tflite;
  struct RClass *_class_tflite_model;
  struct RClass *_class_tflite_interpreter;
  struct RClass *_class_tflite_interpreter_options;
  struct RClass *_class_tflite_tensor;
  ARENA_SAVE;

  _class_tflite = mrb_define_module(mrb, "TfLite");

  _class_tflite_model = mrb_define_class_under(mrb, _class_tflite, "Model", mrb->object_class);
  MRB_SET_INSTANCE_TT(_class_tflite_model, MRB_TT_DATA);
  mrb_define_method(mrb, _class_tflite_model, "initialize", mrb_tflite_model_init, MRB_ARGS_REQ(1));
  mrb_define_module_function(mrb, _class_tflite_model, "from_file", mrb_tflite_model_from_file, MRB_ARGS_REQ(1));
  ARENA_RESTORE;

  _class_tflite_interpreter_options = mrb_define_class_under(mrb, _class_tflite, "InterpreterOptions", mrb->object_class);
  MRB_SET_INSTANCE_TT(_class_tflite_interpreter_options, MRB_TT_DATA);
  mrb_define_method(mrb, _class_tflite_interpreter_options, "initialize", mrb_tflite_interpreter_options_init, MRB_ARGS_NONE());
  mrb_define_method(mrb, _class_tflite_interpreter_options, "num_threads=", mrb_tflite_interpreter_options_num_threads_set, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, _class_tflite_interpreter_options, "add_delegate", mrb_tflite_interpreter_options_add_delegate, MRB_ARGS_REQ(1));

  _class_tflite_interpreter = mrb_define_class_under(mrb, _class_tflite, "Interpreter", mrb->object_class);
  MRB_SET_INSTANCE_TT(_class_tflite_interpreter, MRB_TT_DATA);
  mrb_define_method(mrb, _class_tflite_interpreter, "initialize", mrb_tflite_interpreter_init, MRB_ARGS_ARG(1, 1));
  mrb_define_method(mrb, _class_tflite_interpreter, "allocate_tensors", mrb_tflite_interpreter_allocate_tensors, MRB_ARGS_NONE());
  mrb_define_method(mrb, _class_tflite_interpreter, "invoke", mrb_tflite_interpreter_invoke, MRB_ARGS_NONE());
  mrb_define_method(mrb, _class_tflite_interpreter, "input_tensor_count", mrb_tflite_interpreter_input_tensor_count, MRB_ARGS_NONE());
  mrb_define_method(mrb, _class_tflite_interpreter, "input_tensor", mrb_tflite_interpreter_input_tensor, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, _class_tflite_interpreter, "output_tensor_count", mrb_tflite_interpreter_output_tensor_count, MRB_ARGS_NONE());
  mrb_define_method(mrb, _class_tflite_interpreter, "output_tensor", mrb_tflite_interpreter_output_tensor, MRB_ARGS_REQ(1));
  ARENA_RESTORE;

  _class_tflite_tensor = mrb_define_class_under(mrb, _class_tflite, "Tensor", mrb->object_class);
  MRB_SET_INSTANCE_TT(_class_tflite_tensor, MRB_TT_DATA);
  mrb_define_method(mrb, _class_tflite_tensor, "type", mrb_tflite_tensor_type, MRB_ARGS_NONE());
  mrb_define_method(mrb, _class_tflite_tensor, "name", mrb_tflite_tensor_name, MRB_ARGS_NONE());
  mrb_define_method(mrb, _class_tflite_tensor, "num_dims", mrb_tflite_tensor_num_dims, MRB_ARGS_NONE());
  mrb_define_method(mrb, _class_tflite_tensor, "dim", mrb_tflite_tensor_dim, MRB_ARGS_REQ(1));
  mrb_define_method(mrb, _class_tflite_tensor, "byte_size", mrb_tflite_tensor_byte_size, MRB_ARGS_NONE());
  mrb_define_method(mrb, _class_tflite_tensor, "data", mrb_tflite_tensor_data_get, MRB_ARGS_NONE());
  mrb_define_method(mrb, _class_tflite_tensor, "data=", mrb_tflite_tensor_data_set, MRB_ARGS_REQ(1));
  ARENA_RESTORE;
}

void
mrb_mruby_tflite_gem_final(mrb_state* mrb) {
}

/* vim:set et ts=2 sts=2 sw=2 tw=0: */
