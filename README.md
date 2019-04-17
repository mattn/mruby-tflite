# mruby-tflite

interface to TensorFlow Lite for mruby

## Usage

```ruby
model = TfLite::Model.from_file "xor_model.tflite"
interpreter = TfLite::Interpreter.new(model)
interpreter.allocate_tensors
input = interpreter.input_tensor(0)
output = interpreter.output_tensor(0)
[[0,0], [1,0], [0,1], [1,1]].each do |x|
  input.data = x
  interpreter.invoke
  puts output.data[0].round
end
```

## Requirements

TensorFlow Lite

## License

MIT

## Author

Yasuhiro Matsumoto (a.k.a. mattn)
