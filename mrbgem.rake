MRuby::Gem::Specification.new('mruby-tflite') do |spec|
  spec.license = 'MIT'
  spec.authors = 'mattn'
 
  spec.cc.include_paths << ENV['TENSORFLOW_ROOT'] 
  spec.linker.library_paths << ENV['TENSORFLOW_ROOT'] + "tensorflow/lite/experimental/c/"
  spec.linker.libraries << 'tensorflowlite_c'
end
