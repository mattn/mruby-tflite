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
end
