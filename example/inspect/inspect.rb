#!mruby
#
# Print the tensor layout of a .tflite model. Handy for finding out what shape
# of data a model wants before writing code against it.
#
#   mruby inspect.rb ../xor/xor_model.tflite ../fizzbuzz/fizzbuzz_model.tflite

TYPE_NAMES = {
  0 => 'none', 1 => 'float32', 2 => 'int32', 3 => 'uint8', 4 => 'int64',
  5 => 'string', 6 => 'bool', 7 => 'int16', 8 => 'complex64', 9 => 'int8',
}

def dump(tensor)
  dims = (0...tensor.num_dims).map { |i| tensor.dim(i) }
  type = TYPE_NAMES[tensor.type] || "type #{tensor.type}"
  name = tensor.name || '(no name)'
  puts "    #{name}: #{type}[#{dims.join(', ')}] (#{tensor.byte_size} bytes)"
end

if ARGV.empty?
  puts 'usage: inspect.rb MODEL.tflite...'
else
  ARGV.each do |path|
    model = TfLite::Model.from_file path
    interpreter = TfLite::Interpreter.new(model)
    interpreter.allocate_tensors

    puts path
    puts "  #{interpreter.input_tensor_count} input(s)"
    interpreter.input_tensor_count.times { |i| dump interpreter.input_tensor(i) }
    puts "  #{interpreter.output_tensor_count} output(s)"
    interpreter.output_tensor_count.times { |i| dump interpreter.output_tensor(i) }
  end
end
