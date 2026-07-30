assert('xor') do
  model = TfLite::Model.from_file(TEST_ARGS['model'])
  interpreter = TfLite::Interpreter.new(model)
  interpreter.allocate_tensors
  input = interpreter.input_tensor(0)
  output = interpreter.output_tensor(0)
  [[0, 0], [1, 0], [0, 1], [1, 1]].each do |x|
    input.data = x
    interpreter.invoke
    assert_equal(x[0] ^ x[1], output.data[0].round)
  end
end

assert('tensor index out of range') do
  model = TfLite::Model.from_file(TEST_ARGS['model'])
  interpreter = TfLite::Interpreter.new(model)
  interpreter.allocate_tensors
  assert_raise(ArgumentError) { interpreter.input_tensor(100) }
  assert_raise(ArgumentError) { interpreter.input_tensor(-1) }
  assert_raise(ArgumentError) { interpreter.output_tensor(100) }
  assert_raise(ArgumentError) { interpreter.output_tensor(-1) }
  input = interpreter.input_tensor(0)
  assert_raise(ArgumentError) { input.dim(100) }
  assert_raise(ArgumentError) { input.dim(-1) }
end

assert('invalid arguments raise') do
  assert_raise(ArgumentError) { TfLite::Interpreter.new(nil) }
  assert_raise(ArgumentError) { TfLite::Interpreter.new(1) }
  model = TfLite::Model.from_file(TEST_ARGS['model'])
  assert_raise(ArgumentError) { TfLite::Interpreter.new(model, 1) }
  assert_raise(TypeError) { TfLite::Tensor.new.name }
  interpreter = TfLite::Interpreter.new(model)
  interpreter.allocate_tensors
  input = interpreter.input_tensor(0)
  assert_raise(TypeError) { input.data = ["a", "b"] }
  assert_raise(ArgumentError) { input.data = [1] }
  assert_raise(ArgumentError) { input.data = 1 }
end

assert('gc does not collect objects in use') do
  data = IO.read(TEST_ARGS['model'])
  interpreter = TfLite::Interpreter.new(TfLite::Model.new(data))
  data = nil
  GC.start
  interpreter.allocate_tensors
  input = interpreter.input_tensor(0)
  output = interpreter.output_tensor(0)
  GC.start
  input.data = [1, 0]
  interpreter.invoke
  assert_equal(1, output.data[0].round)
end
