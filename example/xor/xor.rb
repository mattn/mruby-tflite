#!mruby

model = TfLite::Model.from_file "xor_model.tflite"
interpreter = TfLite::Interpreter.new(model)
interpreter.allocate_tensors
input = interpreter.input_tensor(0)
output = interpreter.output_tensor(0)
[[0,0], [1,0], [0,1], [1,1]].each do |x|
  input.data = x
  interpreter.invoke
  puts "#{x[0]} ^ #{x[1]} = #{output.data[0].round}"
end

