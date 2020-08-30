MRuby::Gem::Specification.new('mruby-tflite') do |spec|
  spec.license = 'MIT'
  spec.authors = 'mattn'
  spec.version = '2.3.0'

  add_test_dependency 'mruby-env'
  ENV['MRB_TFLITE_XORMODEL'] = "#{dir}/test/xor_model.tflite"

  if ENV['TENSORFLOW_ROOT']
    spec.cc.include_paths << ENV['TENSORFLOW_ROOT']
    spec.linker.library_paths << ENV['TENSORFLOW_ROOT'] + "tensorflow/lite/experimental/c/"
    spec.linker.libraries << 'tensorflowlite_c'
  else
    file "#{dir}/src/mrb_tflite.c" => __FILE__ do |t|
      FileUtils.mkdir_p build_dir
      Dir.chdir build_dir do
        unless Dir.exists? 'tensorflow'
          sh "git clone https://github.com/tensorflow/tensorflow.git --depth 1 -b v#{version}"
          sh "cd tensorflow; patch -p1 -i #{dir}/tensorflow.patch"
        end
        Dir.chdir 'tensorflow' do
          sh 'bazel build --define tflite_with_xnnpack=true ' \
             '//tensorflow/lite:libtensorflowlite.so ' \
             '//tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so ' \
             '//tensorflow/lite/c:libtensorflowlite_c.so'
        end
      end
    end
    cc.include_paths << "#{build_dir}/tensorflow"
    lib_paths = [
      "#{build_dir}/tensorflow/bazel-bin/tensorflow/lite",
      "#{build_dir}/tensorflow/bazel-bin/tensorflow/lite/delegates/gpu",
      "#{build_dir}/tensorflow/bazel-bin/tensorflow/lite/c",
    ]
    linker.library_paths += lib_paths
    ENV['LD_LIBRARY_PATH'] = "#{ENV['LD_LIBRARY_PATH']}:#{lib_paths.join(':')}"
    linker.libraries << 'tensorflowlite' << 'tensorflowlite_gpu_delegate' << 'tensorflowlite_c'
  end
end
