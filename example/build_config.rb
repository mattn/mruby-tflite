# Build config for running the examples.
#
#   git clone --depth 1 https://github.com/mruby/mruby.git
#   cd mruby
#   MRUBY_CONFIG=../mruby-tflite/example/build_config.rb ./minirake
#   cd ../mruby-tflite/example/xor
#   ../../../mruby/build/host/bin/mruby xor.rb
MRuby::Build.new do |conf|
  conf.toolchain
  conf.gembox 'default'
  conf.gem File.expand_path('..', __dir__)
end
