MRuby::Gem::Specification.new('mruby-tflite') do |spec|
  spec.license = 'MIT'
  spec.authors = 'mattn'
 
  #MRuby.each_target do
  #  cc.include_paths << 
  spec.cc.include_paths << 'c:/dev/tensorflow'
  #spec.linker.library_paths << 'c:/dev/tensorflow/'
  spec.linker.libraries << 'tensorflowlite_c'
end
