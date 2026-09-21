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

* TensorFlow Lite

If the TensorFlow Lite C library is already installed (i.e.
`<prefix>/include/tensorflow/lite/c/c_api.h` and
`<prefix>/lib/libtensorflowlite_c.so`), this gem uses it as-is. `/usr/local`,
`/usr`, `/opt/homebrew` and `/opt/local` are searched by default; set
`TFLITE_PREFIX` to use another prefix.

Otherwise the gem falls back to cloning TensorFlow and building it with bazel,
which takes a long time.

## License

MIT

## Author

Yasuhiro Matsumoto (a.k.a. mattn)
