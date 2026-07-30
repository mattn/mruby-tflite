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
